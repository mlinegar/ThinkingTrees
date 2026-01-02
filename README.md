# ThinkingTrees: Oracle-Preserving Summarization (OPS)

Hierarchical summarization with verifiable information preservation guarantees. Build recursive summarization trees that maintain task-critical information through probabilistic auditing and DSPy-based optimization.

## Quick Start

```bash
# Activate environment
source venv/bin/activate

# Start inference servers
./scripts/start_dual_servers.sh

# Run training pipeline (default task + dataset: RILE scoring on manifestos)
./scripts/run_training_pipeline.sh \
  --output-dir outputs/train_$(date +%Y%m%d_%H%M) \
  --train-samples 100 \
  --optimizer bootstrap_random_search

# Full training example (use GenRM for tournaments, prompt tuning on the same port)
./scripts/run_training_pipeline.sh \
  --output-dir outputs/train_$(date +%Y%m%d_%H%M) \
  --train-samples 100 \
  --val-samples 30 \
  --test-samples 30 \
  --enable-genrm \
  --genrm-port 8001 \
  --opt-model-port 8001 \
  --optimizer bootstrap_random_search \
  --optimizer-budget heavy \
  --n-iterations 2

# Init trees are filtered by prompt token budget (set with --max-init-prompt-tokens)

# Run with generic summarization task (still on manifestos by default)
./scripts/run_training_pipeline.sh \
  --task summarization \
  --output-dir outputs/summarization_test
```

## Architecture

```
ThinkingTrees/
├── config/
│   └── settings.yaml              # Model configs, generation params
├── src/
│   ├── core/                      # Generic building blocks
│   │   ├── data_models.py         # Node, Tree, AuditResult
│   │   ├── documents.py           # DocumentSample, DocumentResult
│   │   ├── llm_client.py          # LLMClient (vLLM/OpenAI)
│   │   ├── signatures.py          # DSPy signatures (generic)
│   │   ├── scorers.py             # ScaleScorer, PairwiseScorer
│   │   ├── summarization.py       # GenericSummarizer, GenericMerger
│   │   ├── batch_processor.py     # Level-wise batch processing
│   │   ├── output_parser.py       # Case-insensitive LLM output parsing
│   │   └── strategy.py            # SummarizationStrategy protocol + registry
│   │
│   ├── datasets/                  # Dataset plugins (where data comes from)
│   │   ├── base.py                # DatasetPlugin protocol
│   │   ├── manifesto.py           # Manifesto dataset
│   │   └── jsonl.py               # Generic JSONL dataset
│   │
│   ├── ops_engine/
│   │   ├── builder.py             # TreeBuilder (bottom-up construction)
│   │   ├── auditor.py             # Probabilistic verification
│   │   ├── bootstrap_loop.py      # Multi-iteration training loop
│   │   ├── scoring.py             # OracleScore, ScoringOracle, create_oracle_scorer
│   │   ├── initialization.py      # Top-down demo seeding
│   │   └── training_framework/
│   │       ├── config.py          # Training configuration
│   │       ├── preference.py      # Preference learning
│   │       ├── preference_types.py # PreferenceDeriver protocol + registry
│   │       ├── genrm_preference.py # GenRM integration
│   │       ├── metrics.py         # MetricBuilder + metric factories
│   │       ├── labeling.py        # LabelingStrategy protocol + registry
│   │       ├── optimizers/        # DSPy optimizer registry
│   │       ├── judges/            # Pairwise comparison judges
│   │       │   ├── base.py        # BaseJudge protocol, JudgeResult
│   │       │   ├── dspy.py        # DSPyJudge (optimizable)
│   │       │   ├── genrm.py       # GenRMJudgeWrapper
│   │       │   └── oracle.py      # OracleJudge
│   │       └── tasks/             # Pluggable task system
│   │           ├── base.py        # AbstractTask, ScaleDefinition
│   │           ├── registry.py    # Task discovery
│   │           ├── scoring.py     # Generic ScoringTask
│   │           └── document_analysis.py # Content preservation (0 to 1)
│   │
│   ├── pipelines/                 # Task/dataset-agnostic pipelines
│   │   └── batched.py             # Batched inference pipeline
│   │
│   ├── tasks/                     # Domain-specific building blocks
│   │   └── manifesto/             # RILE scoring example
│   │       ├── constants.py       # RILE scale bounds
│   │       ├── rubrics.py         # RILE preservation rubric
│   │       ├── data_loader.py     # ManifestoDataset
│   │       ├── dspy_signatures.py # RILEScorer, RILEComparator
│   │       └── summarizer.py      # RILE-specific summarizers
│   │
│   └── training/
│       └── run_pipeline.py        # Main training entry point
│
├── scripts/
│   ├── start_dual_servers.sh      # Start inference servers
│   ├── run_training_pipeline.sh   # Training wrapper
│   └── stop_small_servers.sh      # Server shutdown
│
└── experiments/                   # Experiment scripts
```

## Core Concepts

### Node
The atomic unit of the summarization tree:

```python
@dataclass
class Node:
    id: str                          # Unique identifier
    level: int                       # 0 = leaf, higher = more summarized
    raw_text_span: Optional[str]     # Original text (leaves only)
    summary: str                     # Summary at this node
    left_child: Optional[Node]       # Left subtree
    right_child: Optional[Node]      # Right subtree
    audit_result: AuditResult        # Verification status
```

### Building Blocks Pattern
Tasks are composed from generic building blocks, not hardcoded:

```python
from src.tasks.base import ScoringTask, ScaleDefinition
from src.core import ScaleScorer, GenericSummarizer

# Define your scale
MY_SCALE = ScaleDefinition(
    name="sentiment",
    min_value=-1.0,
    max_value=1.0,
    description="Sentiment score",
)

# Compose a task from building blocks
task = ScoringTask(
    name="sentiment",
    scale=MY_SCALE,
    rubric="Preserve sentiment indicators...",
    predictor_factory=lambda: ScaleScorer(MySentimentSignature),
)
```

Example using RILE building blocks from `src/tasks/manifesto/`:

```python
from src.tasks.base import ScoringTask
from src.tasks.manifesto import (
    RILE_SCALE,                  # ScaleDefinition(-100, +100)
    RILE_PRESERVATION_RUBRIC,   # Domain rubric
    ManifestoDataset,           # Data loading
    RILEScorer,                 # Domain scorer
)

rile_task = ScoringTask(
    name="rile",
    scale=RILE_SCALE,
    rubric=RILE_PRESERVATION_RUBRIC,
    data_loader_factory=lambda: ManifestoDataset(),
    predictor_factory=lambda: RILEScorer(),
)
```

Available building blocks in `src/core/`:
- `ScaleScorer` - Generic DSPy scorer for any bounded scale
- `PairwiseScorer` - Generic pairwise comparison scorer
- `GenericSummarizer` - Configurable summarization module
- `GenericMerger` - Configurable merge module

### Normalization and Metrics
Internal optimization uses normalized 0-1 units even when tasks have a real-world scale:

- DSPy metrics expect higher-is-better in [0, 1]; `OracleScore.score` follows this.
- Tournament preference labels are derived from normalized errors (lower is better), not raw scores.
- Raw task values (e.g., RILE -100 to +100) are preserved for reporting and stored alongside normalized errors.
- Tie margins are expressed in normalized units; use the task scale range to convert raw margins.

For tasks with a scale, normalization follows:
```
normalized_error = abs(predicted - ground_truth) / scale.range
```

### Dataset Plugins
Datasets define where documents come from:

```python
from src.datasets import get_dataset

dataset = get_dataset("manifesto")
samples = dataset.load_samples(limit=100)
```

### OPS Laws (Verified by Auditor)
1. **Sufficiency (C1)**: `oracle(summary) ≈ oracle(original)`
2. **Idempotence (C2)**: `oracle(summarize(S)) ≈ oracle(S)`
3. **Merge Consistency (C3)**: `oracle(merge) ≈ aggregate(oracle(children))`

## CLI Reference (src/training/run_pipeline.py)

### Server Options

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8000 | vLLM port for summarizer/inference |
| `--opt-model-port` | None | Optional prompt-tuning LM (set to GenRM port, e.g. 8001) |

### Data Options

| Flag | Default | Description |
|------|---------|-------------|
| `--train-samples` | 33 | Number of training samples |
| `--val-samples` | 11 | Number of validation samples |
| `--test-samples` | 11 | Number of test samples |
| `--rounds` | 3 | Reserved (currently unused) |

### Concurrency

| Flag | Default | Description |
|------|---------|-------------|
| `--concurrent-docs` | 20 | Documents processed in parallel |
| `--concurrent-requests` | 200 | Concurrent LLM requests |
| `--num-threads` | 64 | Parallel metric evaluations |

### Caching

- vLLM prefix caching (APC) is controlled by `vllm.enable_prefix_caching` in `config/settings.yaml` and is enabled by default in the server scripts.
- DSPy response caching is enabled by default; pass `--no-cache` to disable it for a run.
- Oracle memoization is used during iterative optimization via `create_cached_oracle_metric` (per-run in-memory cache of oracle predictions).
- Oracle pre-caching seeds that cache with predictions for the current trainset by default; pass `--no-precache` to skip it.
- Caching is independent of generation temperature; disable caching if you want maximum variability.

### Optimizer

| Flag | Default | Description |
|------|---------|-------------|
| `--optimizer` | bootstrap_random_search | Optimizer (gepa, bootstrap, bootstrap_random_search, mipro, labeled_fewshot) |
| `--optimizer-budget` | heavy | Budget level for GEPA/MIPRO |
| `--max-metric-calls` | None | Explicit metric-call budget (overrides budget) |

### Iterative Optimization

| Flag | Default | Description |
|------|---------|-------------|
| `--n-iterations` | 1 | Iterations (0=until convergence) |
| `--convergence-threshold` | 0.01 | Early stop threshold |
| `--convergence-patience` | 3 | Early stop patience |
| `--skip-oracle-opt` | False | Skip oracle/scorer optimization |

### GenRM OPS Tree Building

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-genrm` | False | Enable GenRM preference collection |
| `--genrm-port` | 8001 | GenRM server port |
| `--genrm-init-samples` | 8 | Number of OPS trees to build |
| `--genrm-init-candidates` | 4 | Candidates per node in tournament |
| `--max-init-prompt-tokens` | 4000 | Max tokens for init prompts (doc + rubric + instructions) |
| `--max-init-doc-chars` | None | Deprecated alias for `--max-init-prompt-tokens` |
| `--train-comparison-module` | False | Train OPSComparisonModule from preferences |

### Judge Optimization

| Flag | Default | Description |
|------|---------|-------------|
| `--optimize-judge` | False | Optimize GenRM judge prompts (single pass) |
| `--judge-optimization-budget` | light | Judge optimization budget |
| `--use-dspy-strategy` | False | Reserved (currently unused) |
| `--load-optimized-judge` | None | Load a pre-optimized judge |

### Tournament of Tournaments (Iterative Judge Optimization)

| Flag | Default | Description |
|------|---------|-------------|
| `--tournament-of-tournaments` | False | Full ToT loop (build → optimize → repeat) |
| `--tot-max-iterations` | 5 | Max ToT iterations |
| `--tot-convergence-threshold` | 0.01 | ToT convergence threshold |
| `--tot-convergence-patience` | 2 | ToT convergence patience |
| `--tot-samples-per-iteration` | 50 | Samples per ToT iteration |
| `--tot-judge-test-split` | 0.2 | Holdout split for judge accuracy |
| `--tot-shuffle-samples` | True | Shuffle samples each iteration |
| `--tot-random-seed` | 42 | RNG seed for ToT sampling |

### Resume and Output

| Flag | Default | Description |
|------|---------|-------------|
| `--resume` | False | Resume from checkpoints |
| `--output-dir` | required | Output directory |

### Inference Only

| Flag | Default | Description |
|------|---------|-------------|
| `--load-scorer-path` | None | Load scorer module and skip optimization |
| `--inference-only` | False | Run inference only (requires scorer path) |

### Scale Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--scale-min` | -100.0 | Minimum score value |
| `--scale-max` | 100.0 | Maximum score value |

### Task/Dataset Selection

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | settings.yaml default | Task plugin (e.g., manifesto_rile, document_analysis) |
| `--dataset` | settings.yaml default | Dataset plugin (e.g., manifesto, jsonl) |
| `--dataset-path` | None | Path for file-based datasets (jsonl) |

## Models

| Model | Port | Use Case |
|-------|------|----------|
| Nemotron-30B-FP8 | 8000 | Default inference |
| GenRM-NVFP4-235B | 8001 | Preference scoring |

## Development

```bash
# Run tests
pytest tests/ -v

# Check syntax
python3 -m py_compile src/**/*.py

# View training logs
tail -f outputs/*/training.log
```

## References

- **AGENTS.md**: Quick reference for AI agents
- **doc/architecture.md**: Detailed system design
- **config/settings.yaml**: All configuration options
