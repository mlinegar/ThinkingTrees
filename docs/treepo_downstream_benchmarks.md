# Treepo Downstream Benchmarks

`treepo` provides the small method and task-benchmark contract. ThinkingTrees
owns richer data generators and registers them with that contract.

## Markov

The Markov bridge is `src.ctreepo.treepo_bridge.markov`.

It registers `markov`, which uses the ThinkingTrees Markov changepoint
generator and the `treepo` `oracle` method:

```bash
./venv/bin/python scripts/run_treepo_markov_benchmark.py \
  --config config/treepo/markov_oracle_smoke.yaml \
  --json-out outputs/treepo_markov.json \
  --csv-out outputs/treepo_markov.csv
```

The benchmark produces ThinkingTrees Markov documents, wraps them as tree-like
`eval_data`, and calls:

```python
treepo.methods.run("oracle", {"oracle_name": "markov_changepoint_count", ...})
```

## FNO

`treepo` provides a generic built-in `family="neural_operator"`. It accepts
`operator_kind` names from `neuralop.models` when they fit the dense leaf-grid
surface, and keeps `operator_kind="conv1d"` as a tiny local baseline. The
short `family="fno"` route points to `operator_kind="fno"`. Use these package
paths for neural-operator root-score fitting and backend comparisons.

The ThinkingTrees bridge is `src.ctreepo.treepo_bridge.fno`. It preserves the
package default and registers the richer ThinkingTrees `FNOFamily` as
`family="thinkingtrees_fno"`:

```python
from src.ctreepo.treepo_bridge import THINKINGTREES_FNO_FAMILY, register_fno_family

family_name = register_fno_family()
assert family_name == THINKINGTREES_FNO_FAMILY

result = treepo.methods.run("fit", {
    "family": family_name,
    "train_data": trees,
    "eval_data": trees,
    "backend_config": {
        "fno_config": fno_config,
        "embedding_client": embedding_client,
        "output_dir": "outputs/fno_fit",
    },
})
```

Use `register_treepo_bridges()` to register all ThinkingTrees adapters currently
provided for `treepo`.

The same pattern registers `lda` and `cardinality`: keep data/task logic in
ThinkingTrees, register a native task benchmark adapter, and delegate execution
to the `treepo` contract.
