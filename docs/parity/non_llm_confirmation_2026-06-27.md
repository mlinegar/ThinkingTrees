# Non-LLM Confirmation, 2026-06-27

These runs recreate larger non-LLM checks for the current simplified surfaces.
They are confirmation-sized GPU runs, not publication-scale reruns.

## Artifacts

Root: `outputs/parity_memos/20260627_confirmation/`

| domain | artifact |
| --- | --- |
| Markov | `markov_neural_operator/neural_operator_markov_compare.json` |
| HLL / cardinality | `hll_merge_learning.json`, `hll_merge_learning.csv` |
| LDA | `lda.json`, `lda.csv`, `lda_sklearn_baseline.json` |

## Commands

```bash
CUDA_VISIBLE_DEVICES=1 ./venv/bin/python   /home/mlinegar/treepo/examples/methods/run_neural_operator_markov_compare.py   --config examples/parity/markov_neural_operator_confirmation.toml   --output-dir outputs/parity_memos/20260627_confirmation/markov_neural_operator

./venv/bin/treepo-bench run hll-merge-learning   --config examples/parity/hll_merge_learning_confirmation.yaml   --json-out outputs/parity_memos/20260627_confirmation/hll_merge_learning.json   --csv-out outputs/parity_memos/20260627_confirmation/hll_merge_learning.csv

./venv/bin/python scripts/run_treepo_lda_benchmark.py   --config examples/parity/lda_confirmation.yaml   --json-out outputs/parity_memos/20260627_confirmation/lda.json   --csv-out outputs/parity_memos/20260627_confirmation/lda.csv

./venv/bin/python examples/parity/lda_sklearn_comparator.py   --preset confirmation   --output outputs/parity_memos/20260627_confirmation/lda_sklearn_baseline.json
```

## Results

Markov:

| operator | n | MAE | Pearson |
| --- | ---: | ---: | ---: |
| FNO | 64 | 2.8323 | 0.4298 |
| Conv1D | 64 | 3.0196 | 0.3640 |

HLL / cardinality:

| precision | train docs | HLL RMSE | learned RMSE | learned / HLL floor |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 64 | 0.1168 | 0.5560 | 4.2766 |
| 6 | 256 | 0.1168 | 0.5235 | 4.0271 |
| 8 | 64 | 0.0572 | 0.3047 | 4.6880 |
| 8 | 256 | 0.0572 | 0.1471 | 2.2634 |
| 10 | 64 | 0.0247 | 0.0247 | 0.7600 |
| 10 | 256 | 0.0247 | 0.0247 | 0.7600 |

LDA:

| check | metric | value |
| --- | --- | ---: |
| treepo | exact root count L1 | 0.0000 |
| treepo | exact root pi L1 | 0.0000 |
| treepo | tree sketch pi L1 to full | 0.1021 |
| treepo | full-doc operator pi L1 to full | 0.7590 |
| scikit-learn | topic cosine mean after alignment | 0.9898 |
| scikit-learn | pi L1 to true mean | 0.1857 |

## Reading

The larger checks recreate the non-LLM evidence from commands in the repo.
They confirm the simplified interfaces run at larger sizes and make the
remaining gaps visible: Markov FNO is better than Conv1D in this check but is
not a publication rerun; HLL reaches the floor at p=10 but not at p=6/p=8;
LDA exact/additive structure remains exact while scikit-learn improves with
more data.
