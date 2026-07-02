# Paper Simulation Map

This file is the current paper-facing map for the formal simulation reruns. It is meant to answer two questions:

1. Which simulation roots are canonical for the paper?
2. Which canonical suite report command turns each root into the paper-facing draft artifact?

The current canonical rerun root is:

- `outputs/formal_reruns_20260310_062551`

## Unified Smoke Examples

For a small deterministic paper-facing artifact bundle that exercises the same
contract-driven tree framework across text, symbolic, and learned-state
settings, run:

```bash
venv/bin/python scripts/run_paper_unified_examples.py --output-dir outputs/paper_unified_examples
```

The command writes a top-level `manifest.json` plus per-contract summaries,
including LabeledTree distillation artifacts for the text contract.

## Core Paper Layers

### 1. Baseline anchor

- Root: `outputs/formal_reruns_20260310_062551/cpu_megasweep`
- Purpose: broad baseline sweep across Markov, Segment-LDA OPS recovery, Segmented-LDA C-TreePO, and mergeable controls
- Paper role: baseline anchor figures and reference diagnostics
- Canonical suite report:
  - `venv/bin/python -m src.ctreepo.cli sim suite cpu-megasweep report --output-root <root>`

### 2. Buildout layer

- Root: `outputs/formal_reruns_20260310_062551/simulation_buildout`
- Purpose: the focused "why the method works / where it breaks" layer
- Paper role: buildout/stress section
- Expected sub-roots:
  - `hard_regimes`
  - `estimator_stress`
  - `guidance_frontier`
  - `ipw_expanded`
- Canonical suite report:
  - `venv/bin/python -m src.ctreepo.cli sim suite simulation-buildout report --output-root <root>`

### 3. Clean core publication slice

- Root: `outputs/formal_reruns_20260310_062551/identifiable_zero_longrun_clean`
- Purpose: compact cross-family oracle-equivalence story for the main paper
- Paper role: the main clean cross-family publication figures
- Canonical suite report:
  - `venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-publication report --profile publication_clean --output-root <root>`
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
- Canonical suite report:
  - `venv/bin/python -m src.ctreepo.cli sim suite publication-ctreepo report --output-root <root>`

## Appendix / Robustness Layers

### 6. Learnability

- Expected root: `outputs/formal_reruns_20260310_062551/identifiable_zero_learnability`
- Purpose: appendix-quality learnability story
- Canonical suite report:
  - `venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-learnability report --output-root <root>`

### 7. Neural operator overnight

- Root: `outputs/formal_reruns_20260310_062551/identifiable_zero_neural_operator_v2`
- Purpose: operator-capacity / operator-guidance robustness
- Canonical suite report:
  - `venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-neural-operator report --output-root <root>`

### 8. LDA leaf-noise progression

- Expected root: `outputs/formal_reruns_20260310_062551/identifiable_zero_lda_leafnoise`
- Purpose: appendix-style LDA degradation / leaf-noise story
- Canonical suite report:
  - `venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-lda-leafnoise report --output-root <root>`

### 9. DTM-LDA

- Expected root: `outputs/formal_reruns_20260310_062551/identifiable_zero_dtm_lda`
- Purpose: additional appendix / robustness coverage
- Canonical suite report:
  - `venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-dtm-lda report --output-root <root>`

### 10. Markov supervision-recovery parity grid

- Root: `outputs/markov_supervision_recovery_parity_grid_<timestamp>`
- Purpose: tree-vs-FNO parity evidence under fair recipes, sweeping leaf
  granularity from 8 leaves (16 tokens/leaf) to 1 leaf (128 tokens/leaf =
  single-leaf regime on 128-token documents)
- Paper role: Section 7.5 ("Tree-FNO Parity in the Single-Leaf Regime")
- Key properties:
  - Five recipe ablation ladder: `historical_replay`, `optimization_fairness`,
    `capacity_fairness`, `matched_root`, `fairfno_matched_root`
  - `fixed_leaf_tokens=128` is the true single-leaf coincidence point
  - Exact-collapse candidates verify tree=FNO when leaf spans the full document
  - Uses historical `official_fno/full100` reference (seeds `[0, 1]`) as anchor
  - Does NOT rerun FNO; tree-side only
- Relationship to other layers: complementary to the formal reruns root
  (Layers 1-9 test mechanism claims; this tests practical competitiveness)
- Prepared runner:
  - `scripts/run_markov_supervision_recovery_parity_grid.py`
- Report merger:
  - `scripts/report_markov_cohort_compare.py --parity-grid-root <root>`

### 11. Markov hazard-panel paper tradeoff

- Prepared data root: `outputs/markov_hazard_panel_data_seed0`
- Raw bundles:
  - `outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json`
  - `outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t2048/seed_0/base_bundle.json`
- Prepared tree/FNO cache root:
  - `outputs/_prepared_data/markov_hazard_panels`
- Purpose: paper-facing mixed-regime Markov simulation where aggregate
  performance is reported alongside condition-wise and mean-guess diagnostics.
- Paper role: central anti-mean Markov tradeoff surface. Single-cell
  `recoverable_v5_*` and `structural_core_v2_*` runs remain ablations.
- Panels:
  - `paper_hazard_panel_v1_t128`: 128-token docs, four equal hazard cells
  - `paper_hazard_panel_v1_t2048`: 2048-token composition stress, same axes
    with boundary counts scaled by `sqrt(2048 / 128)`
- Standard train ladder: `[1024, 4096, 10240]`
- Balance contract:
  - train: `2560` docs per condition at `10240`
  - val/test: `256` docs per condition at `1024`
  - train prefixes: `256`, `1024`, `2560` docs per condition at prefixes
    `1024`, `4096`, `10240`
- Current seed-0 mean-guess gaps:
  - `paper_hazard_panel_v1_t128`: `1.4174`
  - `paper_hazard_panel_v1_t2048`: `7.8958`
- Preparation:
  - `python scripts/prepare_markov_hazard_panel_data.py`
- Dry-run:
  - `python scripts/run_markov_optimization_tradeoff_pipeline.py --config config/markov/tradeoff_pipeline.hazard_panel_paper.toml --plan-only`
- Detailed doc:
  - `docs/markov_hazard_panels.md`

### Appendix F — Classical-HLL parity (Proposition 1 empirical companion)

- Root: `outputs/classical_parity/hll/`
- Paper section: `paper/ctreepo/appendix/F_classical_parity.tex`, referenced from `sections/04_theory.tex:100`
- Purpose: empirically exhibit Proposition 1's reduction — classical HLL through TreePO's tree reduction equals classical HLL through the flat reference pipeline.
- Lean anchor: [`HLLIdempotence.lean`](../lean3/FormalProofs/OPT/HLLIdempotence.lean) (see also `ClassicalSketchLocalLaws.lean` for the cross-sketch summary).
- Reproduction: `python scripts/run_classical_parity_benchmark.py --out outputs/classical_parity` then `bash paper/ctreepo/tables/make_tables.sh`. Full recipe in [`docs/classical_parity_benchmark.md`](classical_parity_benchmark.md).
- Routes every cell — native / DataSketches, flat / tree, analytic-oracle / HLL-reference-oracle — through `fit()` so the comparison is a single CSV join.

## Diagnostic-Only Layers

### 10. LDA tree recovery production suite

- Root: `outputs/formal_reruns_20260310_062551/lda_tree_recovery_production`
- Purpose: dedicated LDA tree-recovery experiment
- Paper role: diagnostic only; not the paper-facing local-law/learnability LDA counterpart
- Canonical suite report:
  - `venv/bin/python -m src.ctreepo.cli sim suite lda-tree-recovery-progress report --output-root <root>`

## Objective / IPW Contract

The current reruns are intended to reflect the realized optimized objective, not report-time defaults.

- Markov and LDA local-law runs now serialize the realized objective payload.
- Estimator-aware variants such as `configured_objective_ht` and `configured_objective_hajek` are first-class.
- Report layers should consume realized weights and estimator-aware objective fields whenever they exist.
- Oracle observation design is represented row-by-row by `observed` and the logged `propensity`.
  Deterministic yes is `observed=true, propensity=1`; deterministic no is
  `observed=false, propensity=0`; randomized labels record the known inclusion
  probability used by the sampling design. IPW recovery of a dense-oracle
  population requires positive inclusion probability for the rows in that
  estimand; zero-propensity rows are valid proxy-only rows, not hidden oracle
  observations.
- Public summaries describe the design under `oracle_observation_design`, with
  optional design parameters nested there only when active. They should not emit
  top-level observation-mode or unused sampling-rate defaults.
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

- resolves canonical suites from `src/ctreepo/sim/suite/registry.py`
- runs suite `report` commands where data exists
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
