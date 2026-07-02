# Markov Hazard Panels

This note defines the paper-facing Markov DGP vocabulary used to avoid
benchmarks that can be solved by guessing one global mean transition count.

The generator backend is still `hazard_topic`: a hidden topic stays fixed at
each token unless it switches with probability `hazard_switch_prob`, and
tokens are emitted from disjoint topic palettes. The panel layer only chooses
named mixtures of hazard conditions and records condition metadata.

## Axes

- **Document length**: `t128` is the compact diagnostic setting; `t2048` is
  the composition-stress setting.
- **Switch density**: lower-switch cells target fewer boundaries; higher-switch
  cells target more.
- **Regime count**: `r4` and `r12` separate few-topic and many-topic cases.
- **Condition mixture**: paper panels stratify train/val/test splits across
  conditions so aggregate MAE cannot hide a global-mean predictor.

## Paper Panels

- `paper_hazard_panel_v1_t128`: equal mixture of `r4_p031`, `r12_p031`,
  `r4_p079`, and `r12_p079` at 128 tokens.
- `paper_hazard_panel_v1_t2048`: the same regime/switch axes at 2048 tokens,
  with expected boundary counts scaled by `sqrt(2048 / 128)`.

Compatibility aliases remain valid. For example, `recoverable_v5_t128`,
`recoverable_v5_t2048`, and `structural_core_v2_t128::r12_p079` resolve to
single-condition panels or conditions.

## Diagnostics

Generated panel bundles store `hazard_panel_id`, condition IDs for every split,
and per-condition counts in `MarkovOPSDataBundle.metadata`. Run summaries then
report:

- global mean-baseline MAE;
- condition mean-baseline MAE;
- the mean-guess gap between those two;
- condition-wise model MAE when predictions are evaluated with condition
  metadata.

The paper-facing check is simple: a valid mixed-panel result should beat the
global mean baseline and should not only win on the easiest condition.

## Paper Dataset Workflow

The canonical paper data are generated once from the hazard-panel registry,
then reused by training/evaluation jobs through `base_bundle.json` and the
prepared tree/FNO cache. This avoids regenerating slightly different corpora
for each run and makes train-prefix comparisons meaningful.

Default preparation command:

```bash
source venv/bin/activate
python scripts/prepare_markov_hazard_panel_data.py
```

Defaults:

- panels: `paper_hazard_panel_v1_t128`, `paper_hazard_panel_v1_t2048`
- split sizes: `10240` train, `1024` val, `1024` test
- seed: `0`
- train prefixes prepared for reporting: `1024`, `4096`, `10240`
- raw bundles: `outputs/_bundles/markov_hazard_panels/{panel_id}/seed_0/base_bundle.json`
- prepared caches: `outputs/_prepared_data/markov_hazard_panels/{panel_id}/prepared_*`
- manifest/report: `outputs/markov_hazard_panel_data_seed0/manifest.json` and
  `outputs/markov_hazard_panel_data_seed0/report.md` when using the fixed
  output directory from the current paper prep.

The prep script writes:

- one raw `MarkovOPSDataBundle` per panel;
- one prepared cache per panel containing `train_fno_docs.json`,
  `val_fno_docs.json`, `test_fno_docs.json`, leaf/internal ordering files, and
  cache `metadata.json`;
- a human-readable report with split sizes, condition counts, corpus
  signatures, root-count histograms/quantiles, and mean-guess diagnostics.

## Prefix-Balanced Ordering

The panel builder generates each condition independently, shuffles within that
condition, then interleaves conditions round-robin. For the four-condition paper
panels this means the full split is balanced and the standard train prefixes
are also balanced:

| Prefix | Count Per Condition |
|---:|---:|
| `1024` | `256` |
| `4096` | `1024` |
| `10240` | `2560` |

For val/test at size `1024`, each condition contributes `256` documents. This
is the key anti-mean property: smaller training rungs see the same mixture as
the full training set instead of accidentally overrepresenting one hazard cell.

The seed-0 paper materialization currently verifies:

| Panel | Train Counts | Val/Test Counts | Test Mean-Guess Gap |
|---|---:|---:|---:|
| `paper_hazard_panel_v1_t128` | `2560` per condition | `256` per condition | `1.4174` |
| `paper_hazard_panel_v1_t2048` | `2560` per condition | `256` per condition | `7.8958` |

The `mean_guess_gap` is
`global_mean_baseline_mae - condition_mean_baseline_mae`. A positive value
means a single global mean predictor is measurably worse than even the
condition-aware mean baseline, so aggregate MAE alone is not enough.

## Paper Tradeoff Config

The compact paper config is:

```bash
python scripts/run_markov_optimization_tradeoff_pipeline.py \
  --config config/markov/tradeoff_pipeline.hazard_panel_paper.toml \
  --plan-only
```

That dry run should plan the two hazard-panel scopes and train-doc ladder
`[1024, 4096, 10240]`. The config points at the generated seed-0 bundles:

- `outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json`
- `outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t2048/seed_0/base_bundle.json`

The same config sets `prepared_data_root =
"outputs/_prepared_data/markov_hazard_panels"` and `prepared_data_allow_create
= false`, so paper jobs fail fast if the cache has not been prepared.

Launch after inspecting the plan:

```bash
python scripts/run_markov_optimization_tradeoff_pipeline.py \
  --config config/markov/tradeoff_pipeline.hazard_panel_paper.toml \
  --output-root outputs/markov_hazard_panel_tradeoff_$(date +%Y%m%d_%H%M%S)
```

Use the generic detached launcher around that command for long overnight runs
if the shell should not own the job lifetime.

## Manual Bundle Smoke

For direct `run_markov_changepoint_ops_count` smoke checks against a mixed panel
bundle, pass enough regime/vocabulary capacity for the largest condition in the
panel:

```bash
python src/ctreepo/sim/cli/run_markov_changepoint_ops_count.py \
  --load-data-bundle outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json \
  --train-docs 40 \
  --val-docs 8 \
  --test-docs 8 \
  --n-regimes 12 \
  --vocab-size 48 \
  --fixed-leaf-tokens 16 \
  --exact-family exact \
  --device cpu \
  --json-summary /tmp/markov_hazard_panel_smoke_summary.json
```

The tradeoff pipeline applies these mixed-panel capacity overrides
automatically through the hazard-panel scope, but the standalone CLI does not
infer them from `--load-data-bundle`.

## Contextual-Sufficiency Controls

The same saved bundles can feed the `sbijax` contextual-sufficiency controls.
This is the preferred way to test exact-sketch and package-summary behavior on
the paper panel instead of on a separately generated toy corpus. The
contextual entrypoint is `ctreepo sim run contextual-sbijax`; the standalone
`ctreepo-contextual-sbijax` console script is also available after reinstalling
editable entrypoints with `python -m pip install -e ".[contextual_sbi]"`.

Status (2026-05-05): the optimize-to-zero thread is resolved by the
repo-owned `learned_local_laws` trainer. Package NASS/NASSS grids are
pre-resolution baselines: they are useful for showing the contrastive-objective
floor, but they are not the exact-zero path. The resolution write-up is
[`contextual_sbijax_optimize_to_zero_resolved_2026-05-05.md`](contextual_sbijax_optimize_to_zero_resolved_2026-05-05.md).

For paper-facing t128 contextual grids, the canonical leaf ladder is
`1, 2, 4, 8, 16, 32, 64`. Do not start the grid at 8: leaves `1, 2, 4` are the
small-item checks that make sure exact-sketch inputs and regime-sequence inputs
are tested before any long-leaf averaging can hide a failure.
The continuation runbook is [`optimize_to_zero.md`](optimize_to_zero.md).

Use the same exact-zero command body for each rung, changing only
`--fragment-len` and the output directory:

```bash
for leaf in 1 2 4 8 16 32 64; do
  XLA_PYTHON_CLIENT_PREALLOCATE=false ctreepo sim run contextual-sbijax \
    --data-source markov \
    --load-data-bundle outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json \
    --sbijax-trainer learned_local_laws \
    --sbijax-method nasss \
    --sbijax-package-theta markov_exact_sketch \
    --sbijax-input-encoding markov_exact_sketch \
    --train-docs 1024 \
    --val-docs 256 \
    --test-docs 256 \
    --fragment-len "$leaf" \
    --context-samples-per-doc 1 \
    --response-signature-contexts 16 \
    --response-signature-slices 8 \
    --embedding-dim 32 \
    --state-dim 25 \
    --hidden-dim 128 \
    --learning-rate 0.0003 \
    --n-iter 1000 \
    --batch-size 128 \
    --local-law-weight 1.0 \
    --local-law-leaf-weight 1.0 \
    --local-law-merge-weight 1.0 \
    --local-law-idempotence-weight 1.0 \
    --local-law-contextual-weight 1.0 \
    --seed 0 \
    --output-root "outputs/optimize_to_zero_laws_grid_t128_next/markov_exact_sketch/leaf_${leaf}"
done
```

To reproduce the pre-resolution package baseline, switch only
`--sbijax-trainer learned_local_laws` back to `--sbijax-trainer package` and
use the shorter baseline training budget from `docs/optimize_to_zero.md`.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false ctreepo sim run contextual-sbijax \
  --data-source markov \
  --load-data-bundle outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json \
  --sbijax-trainer theta_supervised \
  --sbijax-method nasss \
  --sbijax-package-theta markov_exact_sketch \
  --sbijax-input-encoding markov_exact_sketch \
  --train-docs 16 \
  --val-docs 8 \
  --test-docs 8 \
  --fragment-len 8 \
  --context-samples-per-doc 1 \
  --response-signature-contexts 3 \
  --response-signature-slices 2 \
  --n-iter 1 \
  --batch-size 8 \
  --output-root /tmp/contextual_sbijax_hazard_panel_bundle_smoke
```

The probe loader preserves `hazard_panel_id`, split condition IDs, and
condition counts, and it infers the largest mixed-panel palette
(`vocab_size=48`, `n_regimes=12`) from the bundle. In the smoke above, the
`markov_exact_sketch_oracle` diagnostic should report zero contextual error;
nonzero learned error is a property of the selected `sbijax` trainer and
summary network rather than of the hazard-panel corpus.
