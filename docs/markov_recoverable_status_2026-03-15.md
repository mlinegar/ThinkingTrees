# Markov Recoverable Status

Date: 2026-03-15

## Scope

This note summarizes the current state of the recoverable Markov changepoint-count debugging path.

Target benchmark:
- generator profile: `piecewise_disjoint_palette`
- task: full-document changepoint count
- supervision contract for the current exact-recovery push: `root-label only`
- fairness contract:
  - same saved train/val/test bundle across all comparisons
  - same padded full token-sequence arrays and masks for the full-doc neural baselines
  - public API only for iteration and canonical reporting

## Fixed Bundle

Canonical saved bundle:
- `/home/mlinegar/ThinkingTrees/outputs/markov_observed_token_recoverable_v4/markov_data/observed_token_bundle.json`

Split signatures:
- train: `a6fcdb9c39c7bc10b91fba693f39807bfebe97aea38008686b97db025a83732b`
- val: `1ad9838ebb369f4cd55d0a6c6b0187b7fa3878d3607e2699508e5b94afc1f7a0`
- test: `99b1e5f2456ae8467781ca43c62232a63c848cba980a23c2fe896d76d2900c7f`

The code now records shared `full_sequence_input_signatures` for both full-document neural baselines, and the report checks that those signatures match exactly.

## Code State

Implemented:
- Root-label-only objective switches for the two full-doc neural baselines in `/home/mlinegar/ThinkingTrees/src/ctreepo/sim/core/markov_changepoint_ops_count.py`
  - operator objectives:
    - `count_ce_only`
    - `count_ce_plus_scalar_mse`
  - transformer head families:
    - `pooled_count_classifier`
    - `boundary_sum_count_hybrid`
- Exact-match diagnostics are now emitted for both baselines.
- The public run CLI exposes:
  - `--doc-sequence-objective`
  - `--doc-transformer-head-family`
  - `--doc-transformer-layers`
- The observed-token suite now accepts an existing bundle via:
  - `ctreepo sim suite markov-observed-token build|run --bundle-file ...`
- The observed-token report now explicitly labels the recoverable root-only setting as an exact-recovery diagnostic, not the main paper generalization benchmark.

Targeted verification after the refactor:
- `51 passed, 5 warnings`
- command:
  - `pytest tests/tree/test_markov_changepoint_ops_count_simulation.py tests/ctreepo/test_validation_ladder_e2e.py -q --maxfail=1`

## What Is Settled

### Solvability witness

On this exact fixed bundle, the linear control is still essentially exact:

| Method | Root MAE |
|---|---:|
| `ridge bigram` | `2.643e-14` |
| `undersupported` | `0.59375` |

Interpretation:
- the token sequence contains enough information to recover the changepoint count
- there is no information barrier in this recoverable setting
- if the full-sequence neural baselines miss, that is now a training / architecture issue

### Best historical non-root-label-only operator result

This is no longer admissible for the current exact-recovery claim, but it is useful context.

Old endpoint-aux operator best:
- file: `/home/mlinegar/ThinkingTrees/outputs/markov_observed_token_recoverable_v15_operator_endpointaux_fast/operator_only.json`
- `doc_sequence.root_mae = 0.18359375`

That result used visible-token auxiliary supervision and should not be used as the final recoverable claim.

## Current Root-Label-Only Results

All runs below reused the exact saved bundle above via the public API:
- `python -m src.ctreepo.cli sim run markov-ops-count --load-data-bundle ...`

### Pilot results

| Run | File | Root MAE | Test Exact-Match |
|---|---|---:|---:|
| operator, `count_ce_only`, `128/512`, `128 epochs` | `/home/mlinegar/ThinkingTrees/outputs/markov_exact_recovery_grid_v1/pilots/operator_count_ce_only.json` | `0.33984375` | `0.703125` |
| transformer, `pooled_count_classifier`, `128/512`, `256 epochs`, `4 layers` | `/home/mlinegar/ThinkingTrees/outputs/markov_exact_recovery_grid_v1/pilots/transformer_pooled_classifier.json` | `0.45703125` | `0.6015625` |

### Wider operator wave

| Run | File | Root MAE | Test Exact-Match |
|---|---|---:|---:|
| operator, `count_ce_only`, `256/1024`, `128 epochs` | `/home/mlinegar/ThinkingTrees/outputs/markov_exact_recovery_grid_v1/wave1/operator_count_ce_only_256_1024.json` | `0.27734375` | `0.74609375` |
| operator, `count_ce_plus_scalar_mse`, `128/512`, `128 epochs` | `/home/mlinegar/ThinkingTrees/outputs/markov_exact_recovery_grid_v1/wave1/operator_count_ce_plus_scalar_128_512.json` | `0.35546875` | `0.69921875` |
| operator, `count_ce_plus_scalar_mse`, `256/1024`, `128 epochs` | `/home/mlinegar/ThinkingTrees/outputs/markov_exact_recovery_grid_v1/wave1/operator_count_ce_plus_scalar_256_1024.json` | `0.359375` | `0.6875` |

Current best admissible root-label-only result:
- operator, `count_ce_only`, `256/1024`, `128 epochs`
- file: `/home/mlinegar/ThinkingTrees/outputs/markov_exact_recovery_grid_v1/wave1/operator_count_ce_only_256_1024.json`
- `root_mae = 0.27734375`
- `test_exact_match_rate = 0.74609375`

## In Flight

Still running at the time of this note:
- transformer, `boundary_sum_count_hybrid`, `128/512`, `256 epochs`, `4 layers`
- output target:
  - `/home/mlinegar/ThinkingTrees/outputs/markov_exact_recovery_grid_v1/wave1/transformer_boundary_hybrid_128_512.json`

Additional scaling check now started after the note above:
- 10x-train recoverable suite root-only build:
  - `/home/mlinegar/ThinkingTrees/outputs/markov_observed_token_recoverable_10x_v1`
- split signatures for that 10x bundle:
  - train: `0f4f033f7dbdfc3ca7ff9ae729daf3f3ade1f8e131a1694d88e8d07a32140457`
  - val: `1ad9838ebb369f4cd55d0a6c6b0187b7fa3878d3607e2699508e5b94afc1f7a0`
  - test: `99b1e5f2456ae8467781ca43c62232a63c848cba980a23c2fe896d76d2900c7f`
- important property:
  - `val` and `test` stayed fixed
  - only the train corpus changed
- 10x operator run currently in flight:
- `/home/mlinegar/ThinkingTrees/outputs/markov_exact_recovery_grid_v1/tenx/operator_count_ce_only_256_1024.json`

That operator-only 10x run has now finished:
- file:
  - `/home/mlinegar/ThinkingTrees/outputs/markov_exact_recovery_grid_v1/tenx/operator_count_ce_only_256_1024.json`
- config:
  - operator
  - `count_ce_only`
  - `256/1024`
  - `128 epochs`
  - `train_docs = 10240`
- result:
  - `doc_sequence.root_mae = 0.1484375`
  - `doc_sequence test_exact_match_rate = 0.85546875`

So 10x more training data helps a lot:
- 1x best admissible operator: `0.27734375`
- 10x operator on fixed `val/test`: `0.1484375`

But it still does not get near the exact-recovery target.

## Current Interpretation

What looks solid:
- the recoverable Markov setting is valid
- `ridge bigram` remains an exact solvability witness on the same fixed bundle
- the full-doc neural baselines really do receive the exact same token-sequence inputs
- under the clean root-label-only contract, the operator is stronger than the transformer

Update after switching `doc_sequence` to the official `neuraloperator` package FNO backend:
- file:
  - `/home/mlinegar/ThinkingTrees/outputs/markov_official_fno_recoverable_v1/operator_only.json`
- config:
  - fixed recoverable 1x bundle
  - official `neuraloperator` FNO
  - `state_dim=256`
  - `hidden_dim=1024`
  - `128 epochs`
  - root-label only
- result:
  - `doc_sequence.root_mae = 0.10546875`
  - `doc_sequence test_exact_match_rate = 0.89453125`

That is substantially better than the old custom `CTreePOModel` full-doc operator at the same 1x data scale:
- old custom best admissible 1x operator: `0.27734375`
- official FNO 1x result: `0.10546875`

What is not yet true:
- no root-label-only full-sequence neural baseline is close to exact recovery yet
- current best admissible neural result is still far from the target `root_mae <= 1e-3`

The operator is improving with width:
- `128/512` CE-only: `0.3398`
- `256/1024` CE-only: `0.2773`

But the scalar-count auxiliary term is currently hurting rather than helping:
- both `count_ce_plus_scalar_mse` runs are worse than `count_ce_only`

## Most Likely Next Moves

If the in-flight transformer hybrid run is still poor, the clean reading is:
- the current generic full-sequence neural families still miss a simple recoverable case under root-label-only supervision

Reasonable next steps after that:
- keep pushing the operator, not the transformer
- try longer CE-only operator runs before adding more loss terms
- investigate why the full-sequence path is still CPU-heavy even under `--device cuda`
- if needed, move to a more directly boundary-count-oriented official-operator readout while keeping the same root-label-only supervision contract

## Public API Commands Used

Iteration:

```bash
source venv/bin/activate

python -m src.ctreepo.cli sim run markov-ops-count \
  --load-data-bundle outputs/markov_observed_token_recoverable_v4/markov_data/observed_token_bundle.json \
  ...
```

Canonical suite/report path:

```bash
source venv/bin/activate

python -m src.ctreepo.cli sim suite markov-observed-token build \
  --profile recoverable \
  --groups root_only \
  --bundle-file outputs/markov_observed_token_recoverable_v4/markov_data/observed_token_bundle.json \
  ...

python -m src.ctreepo.cli sim suite markov-observed-token run \
  --output-root ...

python -m src.ctreepo.cli sim suite markov-observed-token report \
  --output-root ... \
  --no-emit-pdf
```
