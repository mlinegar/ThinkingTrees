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

### 2. Domain Plugin System (`src/ops_engine/training_framework/domains/`)

The framework uses pluggable domains to support different evaluation tasks.

#### DomainPlugin Protocol (`base.py`)

```python
class DomainPlugin(Protocol):
    """Protocol that all domain implementations must follow."""

    @property
    def name(self) -> str: ...
    @property
    def config(self) -> DomainConfig: ...

    def create_training_source(self, results, **kwargs) -> TrainingDataSource: ...
    def create_metric(self, **kwargs) -> Callable: ...
    def create_rubric(self, **kwargs) -> str: ...
    def create_predictor(self, **kwargs) -> dspy.Module: ...
```

#### Available Domains

| Domain | Scale | Output Type | Use Case |
|--------|-------|-------------|----------|
| `manifesto_rile` | -100 to +100 | Continuous | Political manifesto RILE scoring |
| `summarization` | 0 to 1 | Continuous | Generic summarization quality |

#### Adding a New Domain

```python
from src.ops_engine.training_framework.domains import (
    AbstractDomain, register_domain, ScaleDefinition
)

MY_SCALE = ScaleDefinition(
    name="sentiment",
    min_value=-1.0,
    max_value=1.0,
    description="Sentiment score",
)

@register_domain("sentiment")
class SentimentDomain(AbstractDomain):
    def create_rubric(self, **kwargs) -> str:
        return "Preserve the emotional tone and sentiment..."

    def create_predictor(self, **kwargs) -> dspy.Module:
        return SentimentScorer()
```

### 3. OPS Engine (`src/ops_engine/`)

#### Builder (`builder.py`)
Constructs trees bottom-up through recursive summarization.

```python
class TreeBuilder:
    def __init__(self, summarizer: Summarizer, judge: GenRMJudge = None):
        """
        Args:
            summarizer: Function (content, rubric) -> summary
            judge: Optional judge for tournament selection
        """

    def build_from_text(self, text: str, rubric: str) -> BuildResult:
        """Build tree from raw text."""

    def build_from_chunks(self, chunks: List[TextChunk], rubric: str) -> BuildResult:
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

### 5. Batch Processing (`src/core/batch_processor.py`)

Level-wise batch processing for efficient tree construction.

```python
class LevelWiseBatchProcessor:
    """Process all nodes at a level in batches."""

    def process_level(self, nodes: List[Node], rubric: str) -> List[Node]:
        """Summarize all nodes at current level."""
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

- **Custom Domains**: Implement `DomainPlugin` protocol, register with `@register_domain`
- **Custom Optimizers**: Add to `optimizers/` registry
- **Custom Chunkers**: Implement `chunk_for_ops()` interface
- **Custom Oracles**: Implement `ScoringOracle` protocol

## Deprecated Code

The following modules are deprecated but maintained for backward compatibility:

| Module | Replacement | Notes |
|--------|-------------|-------|
| `src/ops_engine/optimizer.py` | `training_framework/optimizers/` | Use registry-based system |
| `src/ops_engine/training_framework/optimization.py` | `training_framework/optimizers/` | Legacy OracleOptimizer |

Import warnings are emitted when these modules are used.
