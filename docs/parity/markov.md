# Markov

## Reference

The Markov reference is the exact count/transition sketch used by the Markov
simulator and by the Lean-aligned progression tests. The neural-operator paths
compare learned tree reductions to the same full-document target.

## Current Evidence

The stronger evidence is already in the Markov parity ladder:

- `docs/markov_tree_parity_handoff.md` reports exact sketch + exact merge at
  `0.000` root MAE in the progression test.
- The same memo reports a unified tree path at `test_root_mae = 0.0232`,
  compared with an official FNO baseline at `0.0410` for the cited cell.
- `docs/markov_fno_local_law_bridge.md` is careful about scope: Lean covers the
  exact sketch and local-law transport interface; whether FNO + SGD realizes
  the laws is an empirical question.

The package smoke checks that the simplified neural-operator path runs with
both FNO and Conv1D:

```bash
CUDA_VISIBLE_DEVICES=1 ./venv/bin/python   /home/mlinegar/treepo/examples/methods/run_neural_operator_markov_compare.py   --config examples/parity/markov_neural_operator.toml   --output-dir outputs/parity_memos/markov_neural_operator
```

Artifact:
`outputs/parity_memos/20260627_gpu_smoke/markov_neural_operator/neural_operator_markov_compare.json`

Smoke metrics:

| operator | n | MAE | Pearson |
| --- | ---: | ---: | ---: |
| FNO | 8 | 5.0693 | 0.3895 |
| Conv1D | 8 | 5.2168 | -0.2731 |

## Confirmation Run

Command:

```bash
CUDA_VISIBLE_DEVICES=1 ./venv/bin/python   /home/mlinegar/treepo/examples/methods/run_neural_operator_markov_compare.py   --config examples/parity/markov_neural_operator_confirmation.toml   --output-dir outputs/parity_memos/20260627_confirmation/markov_neural_operator
```

Artifact:
`outputs/parity_memos/20260627_confirmation/markov_neural_operator/neural_operator_markov_compare.json`

Confirmation metrics:

| operator | n | MAE | Pearson | mean teacher | mean prediction |
| --- | ---: | ---: | ---: | ---: | ---: |
| FNO | 64 | 2.8323 | 0.4298 | 14.4844 | 14.0824 |
| Conv1D | 64 | 3.0196 | 0.3640 | 14.4844 | 13.2395 |

## Leaf-Size Check

The leaf-size report compares FNO and Conv1D at leaf token counts 8, 16, and
32. It also records the exact-sketch local-law lane, where leaf/merge/idemp
residuals are zero or numerical roundoff across leaf counts 1 through 64.

See [2026-06-27 leaf-size and local-law report](non_llm_leaf_sweep_2026-06-27.md).

## Reading

The smoke proves the current generic neural-operator surface runs the Markov
task across operator kinds. The confirmation run is still not the publication
ladder, but it is a larger fresh reproduction: both operators improve with more
data/training, and FNO leads Conv1D on the same split. The stronger parity
claim still comes from the existing Markov parity ladder.
