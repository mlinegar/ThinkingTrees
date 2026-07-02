# HLL / Cardinality

## Reference

The reference is exact set cardinality. The classical comparator is HLL at each
precision, including the expected HLL relative standard error floor.

## Command

```bash
./venv/bin/treepo-bench run hll-merge-learning   --config examples/parity/hll_merge_learning.yaml   --json-out outputs/parity_memos/hll_merge_learning.json   --csv-out outputs/parity_memos/hll_merge_learning.csv
```

Current artifact:
`outputs/parity_memos/20260627_gpu_smoke/hll_merge_learning.json`

## Current Smoke Metrics

| precision | HLL RMSE | HLL theory RSE | learned RMSE | learned / HLL floor |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 0.1481 | 0.1300 | 0.3426 | 2.6353 |
| 8 | 0.0565 | 0.0650 | 0.0565 | 0.8693 |

## Confirmation Run

Command:

```bash
./venv/bin/treepo-bench run hll-merge-learning   --config examples/parity/hll_merge_learning_confirmation.yaml   --json-out outputs/parity_memos/20260627_confirmation/hll_merge_learning.json   --csv-out outputs/parity_memos/20260627_confirmation/hll_merge_learning.csv
```

Artifact:
`outputs/parity_memos/20260627_confirmation/hll_merge_learning.json`

Confirmation metrics:

| precision | train docs | HLL RMSE | HLL theory RSE | learned RMSE | learned / HLL floor |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 64 | 0.1168 | 0.1300 | 0.5560 | 4.2766 |
| 6 | 256 | 0.1168 | 0.1300 | 0.5235 | 4.0271 |
| 8 | 64 | 0.0572 | 0.0650 | 0.3047 | 4.6880 |
| 8 | 256 | 0.0572 | 0.0650 | 0.1471 | 2.2634 |
| 10 | 64 | 0.0247 | 0.0325 | 0.0247 | 0.7600 |
| 10 | 256 | 0.0247 | 0.0325 | 0.0247 | 0.7600 |

## Leaf-Size And Local-Law Check

The leaf-size report uses the completed throughput-aware HLL artifacts for the
local-law table: explicit leaf/merge/idemp residuals by fragment length from
the JAX grid, plus sampled-node carrier rows from the high-batch
`bs8192/eval16k` runs.

See [2026-06-27 leaf-size and local-law report](non_llm_leaf_sweep_2026-06-27.md).

## Reading

The p=8 smoke reaches the HLL floor in the tiny setting, and the confirmation
run reaches the floor at p=10. The p=6 and p=8 confirmation learned cells
remain above the HLL floor, although p=8 improves substantially with more train
docs. The output includes doc, leaf, and token weighting views, while scalar
headline rows use the doc view.
