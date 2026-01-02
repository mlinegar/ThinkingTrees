# ThinkingTrees Architecture

## System Overview

ThinkingTrees implements Oracle-Preserving Summarization (OPS), a hierarchical approach to document summarization that maintains verifiable information preservation guarantees through probabilistic auditing and DSPy-based optimization.

## Core Components

### 1. Data Models (`src/core/data_models.py`)

#### Node
The atomic unit of the summarization tree.

```python
@dataclass
class Node:
    # Identity
    id: str                              # UUID for tracking
    level: int                           # Tree depth (0 = leaf)

    # Content
    raw_text_span: Optional[str]         # Original text (leaves only)
    summary: str                         # Summary at this node

    # Structure
    left_child: Optional['Node']         # Left subtree
    right_child: Optional['Node']        # Right subtree
    parent: Optional['Node']             # Parent reference

    # Audit State
    audit_result: AuditResult            # Verification status

    # Properties
    @property
    def is_leaf(self) -> bool            # True if no children
    @property
    def children(self) -> List[Node]     # [left_child, right_child] if present
```

**Invariants:**
- Leaves have `raw_text_span` set, internal nodes have `None`
- `level == 0` iff node is a leaf
- `left_child is None` iff `right_child is None` (both or neither)
- Parent references form a valid tree (no cycles)

#### Tree
Container for the complete summarization structure.

```python
@dataclass
class Tree:
    root: Node                           # Tree root (final summary)
    rubric: str                          # Information preservation criteria
    metadata: dict                       # Source doc info, timestamps

    # Properties
    @property
    def height(self) -> int              # Maximum depth
    @property
    def node_count(self) -> int          # Total nodes
    @property
    def leaves(self) -> List[Node]       # All leaf nodes
```

### 2. Task System (`src/ops_engine/training_framework/tasks/`)

The framework uses a **building blocks** approach where generic components are composed
to create domain-specific tasks. The core `ScoringTask` is fully configurable via constructor.

#### Core Building Blocks (`src/core/`)

| Component | Location | Purpose |
|-----------|----------|---------|
| `ScaleScorer` | `src/core/scorers.py` | Generic DSPy scorer for any bounded scale |
| `PairwiseScorer` | `src/core/scorers.py` | Generic pairwise comparison scorer |
| `GenericSummarizer` | `src/core/summarization.py` | Configurable summarization module |
| `GenericMerger` | `src/core/summarization.py` | Configurable merge module |
| `SummarizationResult` | `src/core/summarization.py` | Result dataclass with compression stats |
| `create_oracle_scorer` | `src/ops_engine/scoring.py` | Factory for oracle scorer functions |

#### ScoringTask - The Generic Task

```python
from src.ops_engine.training_framework.tasks import ScoringTask, ScaleDefinition

# Define your scale
MY_SCALE = ScaleDefinition(
    name="sentiment",
    min_value=-1.0,
    max_value=1.0,
    description="Sentiment score from negative to positive",
    neutral_value=0.0,
)

# Create task with all configuration via constructor
task = ScoringTask(
    name="sentiment",
    scale=MY_SCALE,
    rubric="Preserve the emotional tone and sentiment indicators...",
    task_context="Score text on a sentiment scale from -1 (negative) to +1 (positive)",
    predictor_factory=lambda: SentimentScorer(),
    data_loader_factory=lambda: MyDataLoader(),
)
```

#### Available Tasks

| Task | Scale | Use Case |
|------|-------|----------|
| `scoring` | Configurable | Generic scoring (configure via constructor) |
| `document_analysis` | 0 to 1 | Generic content preservation (default) |

#### Example: RILE Task Configuration

The manifesto RILE task demonstrates the building blocks pattern:

```python
from src.ops_engine.training_framework.tasks import ScoringTask, ScaleDefinition
from src.tasks.manifesto import (
    RILE_SCALE,                  # ScaleDefinition(-100, +100)
    RILE_PRESERVATION_RUBRIC,   # Domain-specific rubric
    ManifestoDataLoader,        # Data loading component
    RILEScorer,                 # Domain-specific DSPy scorer
)

# Compose building blocks into a task
rile_task = ScoringTask(
    name="rile",
    scale=RILE_SCALE,
    rubric=RILE_PRESERVATION_RUBRIC,
    data_loader_factory=lambda: ManifestoDataLoader(),
    predictor_factory=lambda: RILEScorer(),
)
```

#### Creating Your Own Task

1. **Define your scale:**
```python
MY_SCALE = ScaleDefinition(
    name="quality",
    min_value=0.0,
    max_value=10.0,
    description="Quality score from 0 (poor) to 10 (excellent)",
)
```

2. **Create a DSPy signature and scorer:**
```python
class QualityScore(dspy.Signature):
    text: str = dspy.InputField()
    task_context: str = dspy.InputField()
    score: float = dspy.OutputField(desc="Quality score 0-10")

# Use generic ScaleScorer or create domain-specific
from src.core import ScaleScorer
scorer = ScaleScorer(QualityScore, score_field="score")
```

3. **Compose into a task:**
```python
task = ScoringTask(
    name="quality",
    scale=MY_SCALE,
    rubric="Preserve indicators of quality...",
    predictor_factory=lambda: ScaleScorer(QualityScore),
)
```

### 3. OPS Engine (`src/ops_engine/`)

#### Builder (`builder.py`)
Constructs trees bottom-up through recursive summarization.

```python
class TreeBuilder:
    def __init__(self, strategy: SummarizationStrategy, config: BuildConfig = None):
        """
        Args:
            strategy: SummarizationStrategy (batched, DSPy, or tournament-wrapped)
            config: BuildConfig (chunking and merge options)
        """

    async def build(self, text: str, rubric: str) -> BuildResult:
        """Build tree from raw text."""

    async def build_from_chunks(self, chunks: List[TextChunk], rubric: str) -> BuildResult:
        """Build tree from pre-chunked text."""
```

**Algorithm:**
```
BUILD_TREE(chunks, rubric):
    1. Create leaf nodes from chunks
    2. level = 0
    3. WHILE len(nodes) > 1:
        a. Pair adjacent nodes: (n0,n1), (n2,n3), ...
        b. For each pair, create parent:
           - summary = SUMMARIZE(left.summary + right.summary, rubric)
           - level = level + 1
        c. nodes = parent nodes
    4. RETURN root node

Note: Tournament selection is implemented via TournamentStrategy, which wraps a base strategy
and performs candidate generation + pairwise elimination internally. TreeBuilder remains
agnostic to tournament logic.
```

#### Auditor (`auditor.py`)
Probabilistic verification of information preservation.

**OPS Laws:**
1. **Sufficiency (C1)**: `oracle(summary) ≈ oracle(original)` for leaves
2. **Idempotence (C2)**: `oracle(summarize(S)) ≈ oracle(S)`
3. **Merge Consistency (C3)**: `oracle(parent) ≈ aggregate(oracle(children))`

```python
class Auditor:
    def audit_tree(self, tree: Tree, sample_rate: float = 0.1) -> AuditReport:
        """Sample nodes and verify OPS laws."""
```

#### Bootstrap Loop (`bootstrap_loop.py`)
Multi-iteration training cycle.

```python
class BootstrapTrainer:
    def train(self, documents, rubric, dspy_module=None) -> BootstrapResult:
        """
        1. Build trees from documents
        2. Run probabilistic audit
        3. Collect training examples from failures
        4. Optimize with DSPy
        5. Repeat until convergence
        """
```

### 4. Training Framework (`src/ops_engine/training_framework/`)

#### Optimizer Registry (`optimizers/`)

```python
from src.ops_engine.training_framework.optimizers import get_optimizer

optimizer = get_optimizer("bootstrap_random_search", config)
compiled = optimizer.compile(student, trainset, metric=metric)
```

**Available Optimizers:**
- `bootstrap_random_search` (default): BootstrapFewShotWithRandomSearch
- `gepa`: GEPA with reflection-based optimization
- `mipro`: MIPROv2 for instruction optimization
- `labeled_fewshot`: Simple labeled few-shot

#### Preference Learning (`preference.py`, `genrm_preference.py`)

```python
class GenRMJudge:
    """Uses GenRM model for pairwise preference judgments."""

    def judge(self, summary_a: str, summary_b: str, context: str) -> PreferencePair:
        """Return which summary better preserves information."""
```

Tournament-of-tournaments optimization uses normalized error (0-1) for preference labels,
while raw task scores are retained for reporting.

#### Training Pipeline Building Blocks

The training framework uses a registry pattern for pluggable components:

| Registry | Location | Available Types |
|----------|----------|-----------------|
| `JudgeRegistry` | `training_framework/judges/` | `dspy`, `genrm`, `oracle` |
| `LabelingStrategy` | `training_framework/labeling.py` | `threshold`, `binary`, `percentile`, `adaptive` |
| `PreferenceDeriver` | `training_framework/preference_types.py` | `judge`, `genrm`, `oracle` |
| `StrategyRegistry` | `src/core/strategy.py` | `batched`, `dspy`, `callable`, `tournament` |

**JudgeRegistry** - Pairwise comparison judges:
```python
from src.ops_engine.training_framework.judges import get_judge, JudgeConfig

config = JudgeConfig(type="genrm", base_url="http://localhost:8001/v1")
judge = get_judge("genrm", config)

result = judge.compare(
    context="Preserve political position",
    original_text="...",
    summary_a="...",
    summary_b="...",
)
print(result.preferred)  # "A", "B", or "tie"
```

**MetricBuilder** - Fluent API for composing metrics:
```python
from src.ops_engine.training_framework.metrics import MetricBuilder

metric = (MetricBuilder()
    .with_oracle(oracle_fn)
    .with_scale(my_scale)
    .with_caching(max_entries=5000)
    .with_feedback()
    .build_metric())
```

**LabelingStrategy** - Error-to-label conversion:
```python
from src.ops_engine.training_framework.labeling import get_labeler

labeler = get_labeler("threshold", threshold_high=0.3, threshold_low=0.1)
label = labeler.label_from_error(error=0.25, scale=my_scale)
```

**StrategyRegistry** - Summarization strategies:
```python
from src.core.strategy import get_strategy

strategy = get_strategy("tournament", base=batched_strategy, judge=my_judge)
summary = await strategy.summarize(content, rubric)
```

### 5. Batch Processing (`src/core/batch_processor.py`, `src/core/batch_orchestrator.py`)

Level-wise batch processing for efficient tree construction.

```python
class BatchTreeOrchestrator:
    """Orchestrate tree building across multiple documents with level-wise batching."""

    async def process_documents(
        self,
        documents: List[Any],
        rubric: str,
        get_text_fn: Callable,
        get_id_fn: Callable,
    ) -> List[BuildResult]:
        """Process all documents with optimal batching, returning BuildResult objects."""
```

**Telemetry:**
```python
@dataclass
class BatchTelemetry:
    total_items: int
    successful: int
    failed: int
    wall_clock_seconds: float
    tokens_used: int
```

### 6. Output Parsing (`src/core/output_parser.py`)

Case-insensitive parsing of LLM outputs.

```python
class NormalizedOutputAccessor:
    """Handle LLM output field casing variations."""

    def get(self, name: str, default: Any = None) -> Any:
        """
        Case-insensitive field access.
        Handles: rile_score, RILE_score, riLE_score, etc.
        """
```

## Data Flow

```
┌─────────────┐     ┌──────────┐     ┌─────────────┐
│  Raw Doc    │────▶│ Chunker  │────▶│ Leaf Nodes  │
└─────────────┘     └──────────┘     └──────┬──────┘
                                            │
                    ┌───────────────────────┘
                    ▼
            ┌───────────────┐
            │   Builder     │◀──── Rubric + Domain
            │ (Summarize)   │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │     Tree      │
            │  (Node root)  │
            └───────┬───────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│   Auditor     │       │ Final Summary │
│  (Verify OPS  │       │   (Output)    │
│   Laws)       │       └───────────────┘
└───────┬───────┘
        │
        ▼ (failures)
┌───────────────┐
│  Bootstrap    │
│   Trainer     │
│ (DSPy Optim)  │
└───────────────┘
```

## Configuration (`config/settings.yaml`)

```yaml
models:
  task_model:
    base_url: "http://localhost:8000/v1"
    model_name: "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
  genrm_model:
    base_url: "http://localhost:8001/v1"
    model_name: "nvidia/Llama-3.1-Nemotron-70B-Instruct"

generation:
  max_tokens: 2000
  temperature: 0.7

training:
  optimizer: "bootstrap_random_search"
  budget: "medium"
  n_iterations: 2
```

## Server Architecture

| Server | Port | Model | GPUs | Purpose |
|--------|------|-------|------|---------|
| Task Model | 8000 | Nemotron-30B-FP8 | 0,1 | Summarization, scoring |
| GenRM | 8001 | GenRM-NVFP4-235B | 2,3 | Preference judgments |

## Error Handling

1. **LLM Output Parsing**: Case-insensitive key matching via `NormalizedOutputAccessor`
2. **LLM Failures**: Retry with exponential backoff
3. **Audit Failures**: Collected as training examples, don't block pipeline
4. **Tree Invariant Violations**: Raise exception (indicates bugs)

## Extension Points

### Building Blocks Approach (Recommended)

The preferred way to create new tasks is by composing generic building blocks:

```python
from src.core import ScaleScorer, GenericSummarizer
from src.ops_engine.training_framework.tasks import ScoringTask, ScaleDefinition

# 1. Define your scale
my_scale = ScaleDefinition(name="my_task", min_value=0, max_value=100)

# 2. Create or use generic components
# scorer = ScaleScorer(MySignature) or create domain-specific

# 3. Compose into a task
task = ScoringTask(name="my_task", scale=my_scale, rubric="...", ...)
```

### Other Extension Points

- **Custom Scorers**: Use `ScaleScorer` with custom signature, or extend for domain-specific logic
- **Custom Summarizers**: Use `GenericSummarizer` with custom signature, or create domain-specific
- **Custom Optimizers**: Add to `optimizers/` registry
- **Custom Chunkers**: Implement `chunk_for_ops()` interface
- **Custom Data Loaders**: Implement loader returning samples with text and labels

### Example Domain Module Structure

See `src/tasks/manifesto/` for a complete example:

```
src/tasks/my_domain/
├── __init__.py           # Export building blocks
├── constants.py          # Scale bounds, field names
├── rubrics.py            # Domain-specific rubrics
├── dspy_signatures.py    # Domain-specific DSPy signatures
├── data_loader.py        # Data loading logic
└── summarizer.py         # Optional domain-specific summarizers
```
