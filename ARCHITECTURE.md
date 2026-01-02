# ThinkingTrees Architecture

## Overview

ThinkingTrees is a generalized framework for document analysis using OPS (Oracle-Preserving Summarization) with batched inference and iterative optimization. The system is task-agnostic and supports various document analysis tasks through a plugin architecture.

## Core Concepts

### 1. Task Plugin System
Tasks define **what** we do with documents (scoring, extraction, classification):
- **Task Plugins** (`src/tasks/`): Define task-specific behavior
- **Examples**: manifesto_rile (political scoring), document_analysis (generic preservation)
- **Interface**: `AbstractTask` provides create_predictor(), create_rubric(), parse_score(), etc.

### 2. OPS Trees
Bottom-up hierarchical summarization trees:
- **Leaf nodes**: Original document chunks
- **Internal nodes**: Merged summaries of children
- **Root**: Final summary representing entire document
- **Purpose**: Compression with information preservation

### 3. Batched Processing
High-throughput parallel document processing:
- **Pipeline** (`src/pipelines/batched.py`): Process multiple documents concurrently
- **Batching**: Pool LLM requests across documents for GPU efficiency
- **Speedup**: ~20x faster than sequential processing

### 4. Optimization Framework
DSPy-based iterative improvement:
- **Optimizer** (`src/training/optimization/`): Improve predictors using training data
- **Metrics**: Task-specific score comparison (e.g., MAE for continuous scales)
- **Output**: Optimized prompts and few-shot examples

Normalization conventions:
- DSPy metrics are higher-is-better in [0, 1] (see `OracleScore.score`).
- Tournament-of-tournaments labels use normalized errors (lower is better).
- Raw task scores are retained for reporting; normalized errors drive optimization.

### 5. Audit System
Quality control through sampling:
- **Auditor** (`src/tree/auditor.py`): Sample tree nodes and verify quality
- **Oracle**: Ground truth or learned approximation
- **Mini-trees**: Generate training data for oracle approximation

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                  Entry Points                           │
│  scripts/run_training_pipeline.sh                       │
│  src/training/run_pipeline.py                           │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                   Task Plugins                          │
│  src/tasks/                                             │
│    - base.py              (AbstractTask interface)      │
│    - registry.py          (task discovery)              │
│    - scoring.py           (generic ScoringTask)         │
│    - manifesto/           (RILE scoring building blocks)│
│    - document_analysis.py (generic preservation)        │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│              Processing Pipeline                        │
│  src/pipelines/batched.py                               │
│    - BatchedDocPipeline    (concurrent processing)      │
│    - BatchedPipelineConfig (configuration)              │
│  src/core/batch_processor.py                            │
│    - AsyncBatchLLMClient   (batched LLM calls)          │
│  src/core/batch_orchestrator.py                         │
│    - BatchTreeOrchestrator (tree-level batching)        │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                 Tree Building                           │
│  src/tree/builder.py                                    │
│    - TreeBuilder           (async-first, strategy)      │
│    - build(), build_sync() (sync convenience functions) │
│  src/core/data_models.py   (Node, Tree structures)      │
│  src/core/strategy.py      (SummarizationStrategy)      │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│              Optimization Engine                        │
│  src/training/                                          │
│    - optimization/         (DSPy-based optimizers)      │
│    - metrics/              (evaluation metrics)         │
│    - preference/           (preference collection)      │
│    - judges/               (pairwise comparison judges) │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                  Audit & Quality                        │
│  src/tree/auditor.py                                    │
│    - Auditor               (probabilistic sampling)     │
│    - ReviewQueue           (human review flagging)      │
│  src/tree/verification.py                               │
│    - TreeVerifier          (OPS law verification)       │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                    Data Layer                           │
│  src/datasets/                                          │
│    - base.py               (DatasetPlugin interface)    │
│    - manifesto.py          (political texts)            │
│    - jsonl.py              (generic JSONL)              │
└─────────────────────────────────────────────────────────┘
```

## Data Flow

### Training Pipeline
```
1. Load Data (Dataset Plugin)
   ↓
2. Build Trees (Batched Pipeline)
   - Chunk documents
   - Build OPS trees (bottom-up)
   - Generate summaries (batched LLM calls)
   - Optional: GenRM tournament selection
   ↓
3. Optimization (Training Framework)
   - Create training examples from trees
   - Optimize predictor using DSPy
   - Evaluate on validation set
   - Iterate until convergence
   ↓
4. Test Evaluation
   - Process test documents
   - Evaluate with trained predictor
   - Report metrics
```

### Inference Pipeline
```
1. Load Documents
   ↓
2. Build Trees (Batched)
   - Use optimized summarizer
   - Parallel processing
   ↓
3. Extract Results
   - Final summaries
   - Task-specific scores
   ↓
4. Optional: Audit
   - Sample tree nodes
   - Verify with oracle
```

## Key Components

### Task Interface (AbstractTask)
```python
class AbstractTask:
    def create_predictor() -> dspy.Module
    def create_rubric() -> str
    def parse_score(response: str) -> float
    def create_trainset(results) -> List[Example]
    def get_task_context() -> str
    def create_oracle_scorer() -> Callable[[str], float]
    def create_preference_labeler() -> Optional[Callable[[PreferencePair, float], Optional[str]]]
```

### Document Processing
```python
# Input: List of documents
samples = [DocumentSample(doc_id, text, reference_score)]

# Output: Results with trees and predictions
results = pipeline.process_batch(samples)
# Each result contains:
#  - tree: OPS tree structure
#  - final_summary: Root summary
#  - estimated_score: Task prediction
#  - reference_score: Ground truth (if available)
#  - tree_stats: Metrics
```

### Configuration
- **Main**: `config/settings.yaml`
- **Tasks**: Define available tasks and defaults
- **Datasets**: Data source configuration
- **LLM**: vLLM server settings, model profiles
- **Optimization**: Iterations, convergence, GenRM settings

## Key Files

### Core
- `src/core/data_models.py` - Node, Tree, AuditResult structures
- `src/core/batch_processor.py` - Batched LLM client
- `src/core/documents.py` - DocumentSample, DocumentResult
- `src/core/strategy.py` - SummarizationStrategy interface
- `src/core/scoring.py` - OracleScore, ScoringOracle

### Processing
- `src/pipelines/batched.py` - Main batched pipeline
- `src/tree/builder.py` - Tree construction (async-first)
- `src/preprocessing/chunker.py` - Document chunking

### Training
- `src/training/run_pipeline.py` - Main training entry point
- `src/training/optimization/` - DSPy optimizers (GEPA, MIPRO, Bootstrap)
- `src/training/metrics/metrics.py` - Evaluation metrics
- `src/training/judges/` - Pairwise comparison judges

### Tasks
- `src/tasks/base.py` - Task interface (AbstractTask, ScaleDefinition)
- `src/tasks/registry.py` - Task discovery
- `src/tasks/manifesto/` - RILE scoring building blocks

## Extension Points

### Adding a New Task
1. Create task class extending `AbstractTask`
2. Implement required methods (create_predictor, create_rubric, etc.)
3. Register with `@register_task(["task_name"])`
4. Add configuration to `config/settings.yaml`

### Adding a New Dataset
1. Create dataset class extending `AbstractDataset`
2. Implement load_documents()
3. Register with `@register_dataset("dataset_name")`

### Custom Optimization
1. Implement optimizer in `src/training/optimization/`
2. Use DSPy's optimizer interface
3. Configure via command-line args or config

## Performance Characteristics

### Batched Processing
- **Throughput**: ~5-20 docs/min (vs. 1 doc/min sequential)
- **Concurrency**: 10-50 concurrent documents
- **Bottleneck**: LLM inference (batching helps maximize GPU utilization)

### Memory
- Trees stored in memory during processing
- Checkpointing for long runs
- Cleanup after each phase

### Optimization
- Iterations: Typically 1-3 for convergence
- Training data: Minimum 4 examples, recommended 20+
- Metrics: Continuous (MAE) or discrete (accuracy)

## Common Workflows

### Train a Task-Specific Model
```bash
./scripts/start_dual_servers.sh  # vLLM servers

./scripts/run_training_pipeline.sh \
  --task manifesto_rile \
  --train-samples 100 \
  --val-samples 30 \
  --test-samples 30 \
  --n-iterations 2 \
  --output-dir outputs/my_run
```

### Use GenRM for Preference Learning
```bash
# Add --enable-genrm for tournament selection
./scripts/run_training_pipeline.sh \
  --task manifesto_rile \
  --enable-genrm \
  --genrm-port 8001 \
  --use-mini-trees
```

### Audit Quality
```python
from src.tree import Auditor, AuditConfig

auditor = Auditor(oracle=my_oracle, config=AuditConfig(budget=10))
violations = auditor.audit(tree, rubric)
```

## Troubleshooting

### Common Issues

**Import Errors**: Ensure you're in project root and dependencies installed
**vLLM Connection**: Check servers running on correct ports (8000, 8001)
**Out of Memory**: Reduce concurrent documents or use smaller model
**Slow Processing**: Increase concurrency or enable speculative decoding

### Debugging

- Set `logging.level: DEBUG` in config
- Use `--verbose` flag with tree building
- Check `outputs/{run_id}/` for checkpoints and logs

## References

- DSPy: https://github.com/stanfordnlp/dspy
- vLLM: https://github.com/vllm-project/vllm
- OPS Framework: See `doc/architecture.md` for detailed theory
