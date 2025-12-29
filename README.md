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
│   ├── core/
│   │   ├── data_models.py         # Node, Tree, AuditResult
│   │   ├── documents.py           # DocumentSample, DocumentResult
│   │   ├── llm_client.py          # LLMClient (vLLM/OpenAI)
│   │   ├── signatures.py          # DSPy signatures
│   │   ├── batch_processor.py     # Level-wise batch processing
│   │   └── output_parser.py       # Case-insensitive LLM output parsing
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
│   │   ├── scoring.py             # OracleScore, ScoringOracle
│   │   ├── initialization.py      # Top-down demo seeding
│   │   └── training_framework/
│   │       ├── config.py          # Training configuration
│   │       ├── preference.py      # Preference learning
│   │       ├── genrm_preference.py # GenRM integration
│   │       ├── optimizers/        # DSPy optimizer registry
│   │       └── domains/           # Pluggable domain system
│   │           ├── base.py        # DomainPlugin protocol
│   │           ├── registry.py    # Domain discovery
│   │           ├── manifesto.py   # RILE scoring (-100 to +100)
│   │           └── summarization.py # Generic quality (0 to 1)
│   │
│   ├── pipelines/                 # Task/dataset-agnostic pipelines
│   │   └── batched.py             # Batched inference pipeline
│   │
│   ├── manifesto/                 # Manifesto-specific components
│   │   ├── data_loader.py         # ManifestoDataset
│   │   ├── signatures.py          # RILEScorer, RILEComparator
│   │   └── evaluation.py          # Domain metrics
│   │
│   └── training/
│       └── run_pipeline.py        # Main training entry point
│
├── scripts/
│   ├── start_dual_servers.sh      # Start inference servers
│   ├── run_training_pipeline.sh   # Training wrapper
│   └── stop_small_servers.sh      # Server shutdown
│
└── experiments/manifesto_rile/    # Experiment scripts
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

### Task Plugins
Tasks define what we do with documents (summarization, scoring, extraction):

```python
from src.tasks import get_task

# RILE score prediction
task = get_task("manifesto_rile")  # Scale: -100 to +100

# Generic summarization quality
task = get_task("summarization")   # Scale: 0 to 1

# Use task-specific components
rubric = task.create_rubric()
predictor = task.create_predictor()
metric = task.create_metric()
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

## Key Pipeline Flags

| Flag | Options | Description |
|------|---------|-------------|
| `--task` | manifesto_rile, summarization | Task plugin (default: manifesto_rile) |
| `--dataset` | manifesto, jsonl | Dataset plugin (default: manifesto) |
| `--domain` | legacy | Legacy alias for `--task` |
| `--optimizer` | bootstrap_random_search, gepa, mipro | DSPy optimizer |
| `--optimizer-budget` | light, medium, heavy | Optimization intensity |
| `--enable-genrm` | - | Use GenRM for preference learning |
| `--n-iterations` | 1, 2+, 0 | Training iterations (0=until convergence) |

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
