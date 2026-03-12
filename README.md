# ThinkingTrees: Oracle-Preserving Summarization (OPS)

Hierarchical summarization with verifiable information preservation guarantees. Build recursive summarization trees that maintain task-critical information through probabilistic auditing and DSPy-based optimization.

## ThinkingTrees And `treepo`

This repository now exposes two public faces:

- `ThinkingTrees`: the full platform for long-document OPS pipelines, task plugins, training, and deployment.
- `treepo`: a focused PyTorch package under [`treepo/`](./treepo) for method-level simulations and reports.

The `treepo` package is the canonical home for the new HyperLogLog streaming/cardinality work. Typical commands:

```bash
cd treepo
pip install -e ".[torch]"
treepo-bench suite cardinality-paper --out-root ../outputs/cardinality --jobs 4
treepo-bench report cardinality --output-root ../outputs/cardinality
```

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

# Full training example (large-model-only path; GenRM/TOT flags are deprecated)
./scripts/run_training_pipeline.sh \
  --output-dir outputs/train_$(date +%Y%m%d_%H%M) \
  --train-samples 100 \
  --val-samples 30 \
  --test-samples 30 \
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

## LawStress Benchmark (MVP)

Generate synthetic local-law stress data for general information extraction (C1/C2/C3-aware) and evaluate staged.
This workflow is not a STEM/math/coding problem generator.

```bash
# 1) Generate benchmark fixtures + pipeline-consumable JSONL
python scripts/generate_manifesto_lawstress.py \
  --output-dir outputs/lawstress_mvp \
  --teacher-base-url http://localhost:8000/v1 \
  --teacher-model /mnt/data/models/nvidia/Qwen3.5-397B-A17B-NVFP4

# 2a) Stage 1: summarization only (small model)
python scripts/eval_manifesto_lawstress.py \
  --records outputs/lawstress_mvp/lawstress_records.jsonl \
  --output-dir outputs/lawstress_eval \
  --mode summarize_only \
  --summarizer-model qwen3.5-4b

# 2b) Stage 2: teacher scoring only (GenRM disabled)
python scripts/eval_manifesto_lawstress.py \
  --records outputs/lawstress_mvp/lawstress_records.jsonl \
  --output-dir outputs/lawstress_eval \
  --mode score_and_judge_only \
  --scorer-model /mnt/data/models/nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --disable-genrm
```

## Teacher Trace Bootstrap (Real Anchors)

Generate training traces from real manifesto anchors:
- sample real manifesto text with known RILE
- generate score-preserving English expansion with the teacher
- produce 2-hop summaries + structured extraction traces

```bash
# Optional: launch 397B teacher on port 8000
./scripts/start_vllm.sh qwen3.5-397b-a17b-nvfp4 --port 8000 --cuda-devices 0,1,2,3

# Generate traces
python scripts/generate_manifesto_teacher_traces.py \
  --output-dir outputs/teacher_trace_bootstrap \
  --train-size 120 \
  --val-size 30 \
  --test-size 30 \
  --teacher-base-url http://localhost:8000/v1 \
  --teacher-model /mnt/data/models/nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --scorer-base-url http://localhost:8000/v1 \
  --scorer-model /mnt/data/models/nvidia/Qwen3.5-397B-A17B-NVFP4
```

Optional: feed generated docs back through the existing JSONL dataset path:

```bash
./scripts/run_training_pipeline.sh \
  --task manifesto_rile \
  --dataset jsonl \
  --dataset-path outputs/teacher_trace_bootstrap/benchmark_docs.jsonl \
  --train-samples 120 \
  --val-samples 30 \
  --test-samples 30
```

## Method Stack v1 (Equal-Maturity UX)

The pipeline now exposes a consistent interface for:

- LLM prompt optimization (Phase 2)
- Embedding proxy heads (`ridge`, `linear_sgd`, `mil_sgd`) (Phase 1.25)
- Neural operators (`CTreePO`, `mergeable_sketch`) (Phase 1.3)
- Generator fine-tuning with LoRA/full-FT toggle (Phase 3.25/3.5)

Quick examples:

```bash
# Embedding proxy with explicit error policy
./scripts/run_training_pipeline.sh \
  --adaptive-embedding-proxy \
  --adaptive-embedding-head-method ridge \
  --embedding-proxy-fail-on-error

# Neural operators + hybrid representation auto-wire
./scripts/run_training_pipeline.sh \
  --train-neural-operators \
  --neural-operators-which both \
  --hybrid-oracle-seeded-ensemble

# Generator fine-tuning (LoRA)
./scripts/run_training_pipeline.sh \
  --train-generator \
  --generator-method dpo \
  --generator-use-lora

# One-command compare (fast-smoke default)
python scripts/run_method_compare.py --output-root outputs/method_compare_smoke
python scripts/report_method_compare.py --manifest outputs/method_compare_smoke/method_compare_manifest.json
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
│   │   ├── strategy.py            # SummarizationStrategy protocol + registry
│   │   ├── scoring.py             # OracleScore, ScoringOracle
│   │   ├── ops_checks.py          # CheckType, CheckConfig
│   │   ├── batch_processor.py     # Async batched LLM client + request pooling
│   │   ├── batch_orchestrator.py  # Global pipelined tree batching across documents
│   │   └── output_parser.py       # Case-insensitive LLM output parsing
│   │
│   ├── tree/                      # Tree building and verification
│   │   ├── builder.py             # TreeBuilder (async-first)
│   │   ├── auditor.py             # Probabilistic verification
│   │   ├── labeled.py             # LabeledTree, LabeledDataset
│   │   └── verification.py        # TreeVerifier, OracleNodeVerifier
│   │
│   ├── training/                  # Training and optimization
│   │   ├── run_pipeline.py        # Main training entry point
│   │   ├── optimization/          # DSPy optimizers (GEPA, MIPRO, Bootstrap)
│   │   ├── preference/            # Preference learning
│   │   ├── judges/                # Pairwise comparison judges
│   │   ├── metrics/               # Evaluation metrics
│   │   └── data_sources/          # Training data sources
│   │
│   ├── tasks/                     # Task plugins
│   │   ├── base.py                # AbstractTask, ScaleDefinition
│   │   ├── registry.py            # Task discovery
│   │   ├── scoring.py             # Generic ScoringTask
│   │   ├── document_analysis.py   # Content preservation (0 to 1)
│   │   └── manifesto/             # RILE scoring building blocks
│   │
│   ├── datasets/                  # Dataset plugins
│   │   ├── base.py                # DatasetPlugin protocol
│   │   ├── manifesto.py           # Manifesto dataset
│   │   └── jsonl.py               # Generic JSONL dataset
│   │
│   ├── pipelines/                 # Task/dataset-agnostic pipelines
│   │   └── batched.py             # Batched inference pipeline
│   │
│   └── preprocessing/             # Document processing
│       └── chunker.py             # DocumentChunker
│
├── scripts/
│   ├── start_dual_servers.sh      # Start inference servers
│   ├── run_training_pipeline.sh   # Training wrapper
│   ├── generate_manifesto_teacher_traces.py  # Real-anchor teacher trace generation
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

### Legacy GenRM/TOT (Deprecated)

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-genrm` | blocked | Deprecated; use local-law bootstrap (teacher scorer + proxy/GEPA), no GenRM |
| `--start-genrm` (wrapper) | blocked | Deprecated; wrapper exits with error |
| `--train-comparison-module` | blocked | Deprecated; wrapper exits with error |

### Judge Optimization (Deprecated GenRM/TOT Path)

| Flag | Default | Description |
|------|---------|-------------|
| `--optimize-judge` | blocked | Deprecated; use local-law bootstrap path |
| `--judge-optimization-budget` | light | Judge optimization budget |
| `--use-dspy-strategy` | False | Reserved (currently unused) |
| `--load-optimized-judge` | None | Load a pre-optimized judge |

### Tournament of Tournaments (Deprecated)

| Flag | Default | Description |
|------|---------|-------------|
| `--tournament-of-tournaments` | blocked | Deprecated; use local-law bootstrap path |
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
| Qwen3.5-397B-A17B-NVFP4 | 8001 | Large teacher/scorer (when launched as second server) |

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
