# Non-LLM 1k/2k Scale, 2026-06-27

These runs recreate larger non-LLM checks at approximately 1k and 2k training
documents. They use the current simplified examples and write machine-readable
metrics to `outputs/parity_memos/20260627_scale_1k_2k/manifest.json`.

## Artifacts

Root: `outputs/parity_memos/20260627_scale_1k_2k/`

| domain | artifact |
| --- | --- |
| Markov 1k | `markov_1k/neural_operator_markov_compare.json` |
| Markov 2k | `markov_2k/neural_operator_markov_compare.json` |
| HLL / cardinality | `hll_merge_learning.json`, `hll_merge_learning.csv` |
| LDA 1k | `lda_1k.json`, `lda_1k.csv`, `lda_sklearn_1k.json` |
| LDA 2k | `lda_2k.json`, `lda_2k.csv`, `lda_sklearn_2k.json` |

## Commands

```bash
CUDA_VISIBLE_DEVICES=1 ./venv/bin/python   /home/mlinegar/treepo/examples/methods/run_neural_operator_markov_compare.py   --config examples/parity/markov_neural_operator_1k.toml   --output-dir outputs/parity_memos/20260627_scale_1k_2k/markov_1k

CUDA_VISIBLE_DEVICES=1 ./venv/bin/python   /home/mlinegar/treepo/examples/methods/run_neural_operator_markov_compare.py   --config examples/parity/markov_neural_operator_2k.toml   --output-dir outputs/parity_memos/20260627_scale_1k_2k/markov_2k

./venv/bin/treepo-bench run hll-merge-learning   --config examples/parity/hll_merge_learning_1k_2k.yaml   --json-out outputs/parity_memos/20260627_scale_1k_2k/hll_merge_learning.json   --csv-out outputs/parity_memos/20260627_scale_1k_2k/hll_merge_learning.csv

./venv/bin/python scripts/run_treepo_lda_benchmark.py   --config examples/parity/lda_1k.yaml   --json-out outputs/parity_memos/20260627_scale_1k_2k/lda_1k.json   --csv-out outputs/parity_memos/20260627_scale_1k_2k/lda_1k.csv

./venv/bin/python scripts/run_treepo_lda_benchmark.py   --config examples/parity/lda_2k.yaml   --json-out outputs/parity_memos/20260627_scale_1k_2k/lda_2k.json   --csv-out outputs/parity_memos/20260627_scale_1k_2k/lda_2k.csv

./venv/bin/python examples/parity/lda_sklearn_comparator.py   --preset 1k   --output outputs/parity_memos/20260627_scale_1k_2k/lda_sklearn_1k.json

./venv/bin/python examples/parity/lda_sklearn_comparator.py   --preset 2k   --output outputs/parity_memos/20260627_scale_1k_2k/lda_sklearn_2k.json
```

## Results

Markov:

| train docs | operator | n eval | MAE | Pearson | mean teacher | mean prediction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | FNO | 256 | 4.1340 | 0.1261 | 18.9727 | 18.6031 |
| 1024 | Conv1D | 256 | 4.1193 | 0.1131 | 18.9727 | 17.7246 |
| 2048 | FNO | 512 | 3.9105 | 0.1139 | 19.3672 | 19.3131 |
| 2048 | Conv1D | 512 | 4.3681 | 0.0959 | 19.3672 | 17.3939 |

HLL / cardinality:

| precision | train docs | HLL RMSE | HLL theory RSE | learned RMSE | learned / HLL floor |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 1024 | 0.1289 | 0.1300 | 0.6496 | 4.9970 |
| 6 | 2048 | 0.1289 | 0.1300 | 0.5978 | 4.5982 |
| 8 | 1024 | 0.0599 | 0.0650 | 0.2578 | 3.9669 |
| 8 | 2048 | 0.0599 | 0.0650 | 0.0707 | 1.0881 |
| 10 | 1024 | 0.0263 | 0.0325 | 0.0263 | 0.8093 |
| 10 | 2048 | 0.0263 | 0.0325 | 0.0263 | 0.8093 |

LDA:

| train docs | check | metric | value |
| ---: | --- | --- | ---: |
| 1024 | treepo | exact root count L1 | 0.0000 |
| 1024 | treepo | exact root pi L1 | 0.0000 |
| 1024 | treepo | tree sketch pi L1 to full | 0.0683 |
| 1024 | treepo | full-doc operator pi L1 to full | 0.2048 |
| 1024 | scikit-learn | topic cosine mean after alignment | 0.9985 |
| 1024 | scikit-learn | pi L1 to true mean | 0.1756 |
| 2048 | treepo | exact root count L1 | 0.0000 |
| 2048 | treepo | exact root pi L1 | 0.0000 |
| 2048 | treepo | tree sketch pi L1 to full | 0.0659 |
| 2048 | treepo | full-doc operator pi L1 to full | 0.1787 |
| 2048 | scikit-learn | topic cosine mean after alignment | 0.9988 |
| 2048 | scikit-learn | pi L1 to true mean | 0.1896 |

## Reading

The 1k/2k scale run gives a clearer picture than the smaller confirmations:

- Markov's generic neural-operator example does not show monotone gains from
  scale under this lightweight config. FNO leads Conv1D at 2k, but this is not
  the stronger Markov publication path.
- HLL reaches the classical floor at p=10, and p=8 gets close at 2k train docs.
  p=6 remains far above the floor.
- LDA scales well: the additive tree sketch stays exact at the root reference,
  its distance to the full-document posterior improves, and scikit-learn
  recovers the synthetic topics almost perfectly after alignment.
