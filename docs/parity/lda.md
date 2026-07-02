# LDA

## Reference

The LDA reference is the known-topic full-document posterior and the exact
additive count sketch. A separate scikit-learn comparator gives a familiar
external baseline on the same synthetic world.

## Commands

Run the `treepo` bridge benchmark:

```bash
./venv/bin/python scripts/run_treepo_lda_benchmark.py   --config examples/parity/lda.yaml   --json-out outputs/parity_memos/lda_gpu.json   --csv-out outputs/parity_memos/lda_gpu.csv
```

Run the scikit-learn comparator:

```bash
./venv/bin/python examples/parity/lda_sklearn_comparator.py
```

Run the larger confirmation:

```bash
./venv/bin/python scripts/run_treepo_lda_benchmark.py   --config examples/parity/lda_confirmation.yaml   --json-out outputs/parity_memos/20260627_confirmation/lda.json   --csv-out outputs/parity_memos/20260627_confirmation/lda.csv

./venv/bin/python examples/parity/lda_sklearn_comparator.py   --preset confirmation   --output outputs/parity_memos/20260627_confirmation/lda_sklearn_baseline.json
```

Current artifacts:

- `outputs/parity_memos/20260627_gpu_smoke/lda_gpu_canonical.json`
- `outputs/parity_memos/20260627_gpu_smoke/lda_sklearn_baseline.json`
- `outputs/parity_memos/20260627_confirmation/lda.json`
- `outputs/parity_memos/20260627_confirmation/lda_sklearn_baseline.json`

## Current Smoke Metrics

Treepo LDA, GPU smoke:

| metric | value |
| --- | ---: |
| exact root count L1 | 0.0000 |
| exact root pi L1 | 0.0000 |
| tree sketch pi L1 to full | 0.1459 |
| tree sketch utility abs to full | 0.0063 |
| full-doc operator pi L1 to full | 0.8351 |
| full-doc operator utility abs to full | 0.0373 |

scikit-learn LDA comparator:

| metric | value |
| --- | ---: |
| topic cosine mean after alignment | 0.7638 |
| topic cosine min after alignment | 0.4140 |
| pi L1 to true mean | 0.5066 |
| utility abs to true mean | 0.0360 |

## Confirmation Metrics

Treepo LDA, GPU confirmation:

| metric | value |
| --- | ---: |
| exact root count L1 | 0.0000 |
| exact root pi L1 | 0.0000 |
| tree sketch pi L1 to full | 0.1021 |
| tree sketch utility abs to full | 0.0110 |
| full-doc operator pi L1 to full | 0.7590 |
| full-doc operator utility abs to full | 0.0881 |

scikit-learn LDA confirmation:

| metric | value |
| --- | ---: |
| topic cosine mean after alignment | 0.9898 |
| topic cosine min after alignment | 0.9841 |
| pi L1 to true mean | 0.1857 |
| utility abs to true mean | 0.0255 |

## Leaf-Size Check

The leaf-size report runs the package LDA bridge at leaf token counts 8, 16,
and 32. The current LDA report emits exact root/reference distances and
tree-sketch/full-doc distances; it does not yet emit direct leaf/merge/idemp
residual columns.

See [2026-06-27 leaf-size and local-law report](non_llm_leaf_sweep_2026-06-27.md).

## Reading

The exact additive reference is exact by construction. In the tiny smoke, the
tree sketch is much closer to the known-topic full-document posterior than the
small neural full-doc operator; the same pattern holds in the larger
confirmation. scikit-learn improves sharply at the larger size and is a useful
outside comparator, but it solves the ordinary unsupervised LDA problem, not
the known-topic tree-sketch task.
