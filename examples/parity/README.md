# Parity Examples

These examples run checks against a clear reference:

- `markov_neural_operator.toml`: generic neural-operator Markov smoke using FNO and Conv1D.
- `markov_neural_operator_confirmation.toml`: larger Markov confirmation with more data and training.
- `hll_merge_learning.yaml`: HLL merge-learning smoke against exact set cardinality and the HLL floor.
- `hll_merge_learning_confirmation.yaml`: larger HLL/cardinality confirmation grid.
- `lda.yaml`: LDA benchmark through the `treepo` bridge.
- `lda_confirmation.yaml`: larger LDA bridge confirmation.
- `lda_sklearn_comparator.py`: external scikit-learn LDA comparator; use `--preset confirmation` for the larger LDA check.
- `manifesto_dspy.md`: DSPy manifesto scoring checks against expert/teacher labels.

The current GPU smoke artifacts are in `outputs/parity_memos/20260627_gpu_smoke/`.
The current larger confirmation artifacts are in `outputs/parity_memos/20260627_confirmation/`.

Scale checks:

- `markov_neural_operator_1k.toml` and `markov_neural_operator_2k.toml`: Markov neural-operator checks at 1k/2k train docs.
- `hll_merge_learning_1k_2k.yaml`: HLL/cardinality grid at 1k/2k train docs.
- `lda_1k.yaml` and `lda_2k.yaml`: LDA bridge checks at 1k/2k train docs.
- `lda_sklearn_comparator.py --preset 1k|2k`: external LDA baselines matched to the scale configs.

The current 1k/2k scale artifacts are in `outputs/parity_memos/20260627_scale_1k_2k/`.

Leaf-size checks:

- `markov_neural_operator_leaf008_1k.toml`, `markov_neural_operator_leaf016_1k.toml`, and `markov_neural_operator_leaf032_1k.toml`: Markov neural-operator checks by leaf token count.
- `lda_leaf008_1k.yaml`, `lda_leaf016_1k.yaml`, and `lda_leaf032_1k.yaml`: LDA bridge checks by leaf token count, with more CPU threads for smaller leaves.
- `hll_leaf064_1k_2k.yaml`, `hll_leaf128_1k_2k.yaml`, and `hll_leaf256_1k_2k.yaml`: package HLL leaf-size checks with GPU enabled and more CPU threads for smaller leaves.

The current leaf-size artifacts are in `outputs/parity_memos/20260627_leaf_sweep/`.
For dense HLL leaf/local-law sweeps, use the completed throughput-aware
`run_hll_sampled_node_rate_grid.py` and JAX local-law grid artifacts. Those
runners balance row work across GPUs and keep high batch/eval sizes, which is
the right path when fragment counts grow.
