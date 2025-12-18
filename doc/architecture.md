# ThinkingTrees Architecture

## System Overview

ThinkingTrees implements Oracle-Preserving Summarization (OPS), a hierarchical approach to document summarization that maintains verifiable information preservation guarantees.

## Core Components

### 1. Data Models (`src/core/data_models.py`)

#### OPSNode
The atomic unit of the summarization tree.

```python
@dataclass
class OPSNode:
    # Identity
    id: str                              # UUID for tracking
    level: int                           # Tree depth (0 = leaf)

    # Content
    raw_text_span: Optional[str]         # Original text (leaves only)
    summary: str                         # Summary at this node

    # Structure
    left_child: Optional['OPSNode']      # Left subtree
    right_child: Optional['OPSNode']     # Right subtree
    parent: Optional['OPSNode']          # Parent reference

    # Audit State
    audit_passed: bool                   # Verification status
    discrepancy_score: float             # Information loss metric [0,1]
    audit_trace: Optional[dict]          # Debug information
```

**Invariants:**
- Leaves have `raw_text_span` set, internal nodes have `None`
- `level == 0` iff node is a leaf
- `left_child is None` iff `right_child is None` (both or neither)
- Parent references form a valid tree (no cycles)

#### OPSTree
Container for the complete summarization structure.

```python
@dataclass
class OPSTree:
    root: OPSNode                        # Tree root (final summary)
    leaves: List[OPSNode]                # Ordered leaf nodes
    rubric: str                          # Information preservation criteria
    metadata: dict                       # Source doc info, timestamps
```

**Properties:**
- `height`: Maximum depth of tree
- `node_count`: Total nodes in tree
- `leaf_count`: Number of leaves
- `audit_failure_rate`: Proportion of failed audits

### 2. Preprocessing (`src/preprocessing/`)

#### Chunker (`chunker.py`)
Splits documents into manageable pieces for leaf node creation.

**Interface:**
```python
class DocumentChunker:
    def __init__(self,
                 max_chunk_chars: int = 2000,
                 overlap_chars: int = 100):
        pass

    def chunk_text(self, text: str) -> List[TextChunk]:
        """Split text into chunks respecting sentence boundaries."""
        pass

    def chunk_file(self, filepath: Path) -> List[TextChunk]:
        """Load and chunk a file."""
        pass
```

**Chunking Strategy (adapted from LangExtract):**
1. Tokenize text into sentences
2. Greedily combine sentences up to `max_chunk_chars`
3. Break at newlines when exceeding limit
4. Never break mid-word

### 3. OPS Engine (`src/ops_engine/`)

#### Builder (`builder.py`)
Constructs the tree bottom-up through recursive summarization.

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

**Parallelization:**
- All nodes at the same level can be summarized concurrently
- Use `asyncio` or thread pool for LLM calls

#### Auditor (`auditor.py`)
Samples nodes to verify information preservation.

**Audit Types:**
1. **Sufficiency Check (Leaves)**: Does summary preserve rubric info from raw text?
2. **Merge Consistency (Internal)**: Does parent summary preserve info from children?

**Sampling Strategy:**
```python
def audit_tree(root: OPSNode, budget: int = 10):
    """
    Sample 'budget' nodes for verification.
    Prioritize:
    - Higher levels (more compression, more risk)
    - Previously failed nodes
    - Random sample for coverage
    """
```

#### Optimizer (`optimizer.py`)
Improves summarization quality based on audit failures.

**Bootstrap Loop:**
1. Collect failed audit examples
2. Human provides corrections → "Golden Examples"
3. Use DSPy BootstrapFewShot to learn from examples
4. Re-compile summarization module
5. Optionally rebuild affected subtrees

### 4. LLM Integration (`src/core/`)

#### LLM Client (`llm_client.py`)
Unified interface for LLM providers.

```python
class LLMClient:
    def __init__(self, provider: str, model: str, **kwargs):
        """Initialize with provider-specific config."""

    def complete(self, prompt: str) -> str:
        """Basic completion."""

    def get_usage(self) -> dict:
        """Token usage statistics."""
```

**Supported Providers:**
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- DeepSeek
- Local models via Ollama

#### Signatures (`signatures.py`)
DSPy signature definitions for structured LLM interactions.

```python
class RecursiveSummary(dspy.Signature):
    """Summarize while preserving rubric-defined information."""
    rubric: str = dspy.InputField()
    content: str = dspy.InputField()
    summary: str = dspy.OutputField()

class OracleJudge(dspy.Signature):
    """Compare two inputs for task-equivalence."""
    rubric: str = dspy.InputField()
    input_a: str = dspy.InputField()
    input_b: str = dspy.InputField()
    is_congruent: bool = dspy.OutputField()
    discrepancy_score: float = dspy.OutputField()
    reasoning: str = dspy.OutputField()
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
            │   Builder     │◀──── Rubric
            │ (Summarize)   │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │   OPS Tree    │
            │  (Root Node)  │
            └───────┬───────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│   Auditor     │       │ Final Summary │
│  (Sample &    │       │   (Output)    │
│   Verify)     │       └───────────────┘
└───────┬───────┘
        │
        ▼ (failures)
┌───────────────┐
│  Optimizer    │
│ (Bootstrap)   │
└───────────────┘
```

## Configuration (`config/settings.yaml`)

```yaml
llm:
  provider: "openai"
  model: "gpt-4"
  max_tokens: 2000
  temperature: 0.3

chunking:
  max_chars: 2000
  overlap: 100

tree:
  max_children: 2  # Binary tree

audit:
  sample_budget: 10
  discrepancy_threshold: 0.1

optimization:
  bootstrap_examples: 5
  max_iterations: 3
```

## Error Handling

1. **LLM Failures**: Retry with exponential backoff, fallback to alternative provider
2. **Chunking Failures**: Log warning, use simple character split as fallback
3. **Audit Failures**: Flag for human review, don't block pipeline
4. **Tree Invariant Violations**: Raise exception, these indicate bugs

## Extension Points

- **Custom Chunkers**: Implement `BaseChunker` interface
- **Custom LLM Providers**: Add to `llm_client.py` provider registry
- **Custom Audit Strategies**: Subclass `BaseAuditor`
- **Visualization Plugins**: Register with `visualization.py`
