# Non-LLM Leaf-Size and Local-Law Report, 2026-06-27

This report separates current package parity checks from theorem-facing local-law artifacts. Markov and LDA package examples report task/reference performance by leaf size. Markov exact-sketch and HLL local-law runs report explicit local-law residuals by leaf size.

Derived tables live in `outputs/parity_memos/20260627_leaf_sweep/`: `markov_task_by_leaf.csv`, `markov_exact_local_law_by_leaf.csv`, `lda_task_by_leaf.csv`, `hll_jax_local_law_by_leaf.csv`, and `hll_sampled_node_by_leaf.csv`.

## Markov: Current Neural-Operator Task Performance

| leaf tokens | operator | eval docs | MAE | Pearson | mean teacher | mean prediction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | conv1d | 256 | 4.1791 | 0.1599 | 19.0898 | 19.0618 |
| 8 | fno | 256 | 4.0570 | 0.1552 | 19.0898 | 19.1752 |
| 16 | conv1d | 256 | 4.3637 | 0.0608 | 19.0898 | 19.8290 |
| 16 | fno | 256 | 4.0525 | 0.1248 | 19.0898 | 19.8255 |
| 32 | conv1d | 256 | 3.9484 | 0.0415 | 19.0898 | 19.2132 |
| 32 | fno | 256 | 4.3970 | 0.0285 | 19.0898 | 20.0480 |

## Markov: Exact-Sketch Local Laws

Source: `outputs/optimize_to_zero_laws_grid_t128/markov_exact_sketch/leaf_*/summary.json`. This is the exact Markov sketch lane; eps values are exact or numerical roundoff.

| leaf count | eps leaf | eps merge | eps idemp | contextual MAE | raw count MAE | first acc | last acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0000 | n/a | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 8 | 0.0000 | 0.0000 | 0.0000 | 9.09e-10 | 0.0000 | 1.0000 | 1.0000 |
| 16 | 0.0000 | 0.0000 | 0.0000 | 2.01e-09 | 0.0000 | 1.0000 | 1.0000 |
| 32 | 2.44e-11 | 2.62e-11 | 0.0000 | 9.79e-10 | 5.81e-08 | 1.0000 | 1.0000 |
| 64 | 2.27e-11 | 2.94e-11 | 0.0000 | 3.60e-09 | 1.08e-07 | 1.0000 | 1.0000 |

## LDA: Current Package Task/Reference Performance

| leaf tokens | train docs | test docs | exact root pi L1 | tree sketch pi L1 to full | tree sketch utility abs to full | full-doc operator pi L1 to full | full-doc utility abs to full |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1024 | 256 | 0.0000 | 0.0609 | 0.0146 | 0.2009 | 0.0468 |
| 16 | 1024 | 256 | 0.0000 | 0.0607 | 0.0145 | 0.2009 | 0.0468 |
| 32 | 1024 | 256 | 0.0000 | 0.0611 | 0.0144 | 0.2009 | 0.0468 |

The current package LDA parity path does not emit per-law residuals. It emits exact root/reference checks and sketch/full-doc distances. The next cleanup is to route LDA through the same local-law row reporter used by Markov/HLL, so LDA can expose eps leaf/merge/idempotence directly.

## HLL: JAX Local-Law Performance

Source: `outputs/hll_jax_local_law_round4_overnight_grid_20260508_065221/grid_summary.csv`, filtered to train_docs=102400, summary_dim=128, estimate_weight=1.0, learned_merge, MLP decoder.

| fragment length | elapsed s | eps leaf | eps merge | eps idemp | register MAE | raw estimate MAE | contextual raw MAE | pred/truth corr |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 208.1 | 0.0071 | 0.0063 | 0.0046 | 0.0071 | 0.9778 | 1.4896 | 0.5728 |
| 32 | 236.2 | 0.0093 | 0.0092 | 0.0049 | 0.0093 | 1.7270 | 2.3116 | 0.5495 |
| 64 | 310.9 | 0.0074 | 0.0080 | 0.0023 | 0.0074 | 3.0005 | 2.5388 | 0.7496 |
| 128 | 370.1 | 0.0067 | 0.0079 | 0.0019 | 0.0067 | 3.7290 | 2.6530 | 0.5366 |
| 256 | 249.4 | 0.0039 | 0.0153 | 0.0021 | 0.0039 | 3.2547 | 1.3931 | 0.1709 |
| 512 | 315.7 | 0.0008 | 0.0129 | 0.0006 | 0.0008 | 1.4840 | 0.4163 | 0.0046 |

## HLL: Sampled-Node Throughput/Carrier Check

Source: completed rows from the `bs8192/eval16k` sampled-node grids, filtered to sampled_node_rate=0.10. These runs use the prior throughput pattern: larger row work is balanced across GPUs and high batch/eval sizes keep smaller leaves from becoming a serial host-side bottleneck.

| leaves | row work | root MAE | root rel MAE | merge state MAE | root merge state MAE | observed rows/doc | effective sample size |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 294912 | 770.2574 | 3.2540 | 2.2742 | 2.2742 | 1.2114 | 897.0381 |
| 4 | 688128 | 46.3749 | 0.1957 | 2.6102 | 3.8727 | 1.5723 | 792.3773 |
| 8 | 1474560 | 196.1955 | 0.7689 | 0.9235 | 1.8353 | 2.4115 | 823.6945 |
| 16 | 3047424 | 46.8112 | 0.2038 | 0.6964 | 1.5252 | 4.0731 | 835.1895 |
| 32 | 6193152 | 144.0686 | 0.5493 | 0.5836 | 1.8542 | 7.1964 | 817.6747 |
| 128 | 25067520 | 48.1756 | 0.2062 | 0.4926 | 2.5923 | 26.5724 | 837.7741 |
| 256 | 50233344 | 42.4541 | 0.1876 | 0.4423 | 4.0297 | 52.1390 | 868.8632 |
| 512 | 100564992 | 48.0681 | 0.2022 | 0.5060 | 3.7490 | 103.0947 | 1636.5569 |

## Reading

- The current Markov FNO/Conv1D package example is a task-performance smoke; the exact-sketch grid is the local-law evidence and stays at zero or numerical roundoff across leaf counts.
- LDA has good present-tense package parity by leaf token count, but its current package report still needs direct local-law residual columns.
- HLL has the richest local-law reporting: explicit eps leaf/merge/idempotence in the JAX grid, plus sampled-node carrier/throughput rows from the high-batch prior work.
- The quick package benchmark is not the right tool for dense HLL leaf sweeps unless it adopts the same throughput pattern: parallel GPU assignment, high batch/eval sizes, and more CPU threads/workers as leaf sizes shrink and fragment counts increase.
