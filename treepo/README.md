# treepo

Method-focused PyTorch package for TreePO / C-TreePO simulations, classical sketch sanity checks, and benchmark reports.

`ThinkingTrees` remains the broader platform/repo. `treepo` is the focused method package for reproducible simulation work, including the new HyperLogLog streaming/cardinality experiments.

## Install

```bash
pip install treepo
```

Optional extras:

```bash
pip install "treepo[sklearn]"   # scikit-learn LDA + RF baselines
pip install "treepo[torch]"     # torch-based learned sketch + cardinality experiments
pip install "treepo[all]"       # all optional extras + dev tools
```

## Public API

```python
from treepo import (
    HLLConfig,
    HyperLogLogSketch,
    CardinalityRecoveryConfig,
    HLLMergeLearningConfig,
    run_cardinality_recovery_experiment,
    run_hll_merge_learning_experiment,
)
```

The public HLL surface is intentionally simple:

- `HyperLogLogSketch.add()` for streaming updates
- `HyperLogLogSketch.update()` for iterable updates
- `HyperLogLogSketch.merge()` for exact sketch merges
- `HyperLogLogSketch.estimate()` for cardinality queries
- `reduce_hll_sketches()` for schedule-invariant tree reduction

## CLI

After install:

```bash
treepo-bench --help
```

## Quick examples

Run the paper-facing cardinality suite:

```bash
treepo-bench suite cardinality-paper --out-root outputs/cardinality --jobs 4
treepo-bench report cardinality --output-root outputs/cardinality
```

Run a single cardinality recovery experiment:

```bash
treepo-bench run cardinality-recovery \
  --config /path/to/cardinality_recovery.json \
  --json-out outputs/cardinality_single.json \
  --csv-out outputs/cardinality_single.csv
```

Run a single HLL merge-learning experiment:

```bash
treepo-bench run hll-merge-learning \
  --config /path/to/hll_merge_learning.json \
  --json-out outputs/hll_merge_single.json \
  --csv-out outputs/hll_merge_single.csv
```

Run a named legacy suite (writes JSON/CSV under the output root):

```bash
treepo-bench suite identifiable-zero-lda-leafnoise --out-root outputs/leafnoise --jobs 8
```

Generate a legacy report from those outputs:

```bash
treepo-bench report lda-leafnoise --output-root outputs/leafnoise
```
