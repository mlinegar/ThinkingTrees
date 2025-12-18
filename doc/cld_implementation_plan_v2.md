# ThinkingTrees Implementation Plan v2

**Synthesized from**: `cld_implementation_plan.md` (comprehensive design) + `cdx_implementation_plan.md` (production engineering)

## Executive Summary

**ThinkingTrees** is a hierarchical document summarization framework implementing Oracle-Preserving Summarization (OPS) from the paper "Hierarchical Summarization with Oracle-Preserving Guarantees". Built on a fork of OmniThink, it transforms the expansion engine (Topic → MindMap → Article) into a **reduction engine** (Document → Tree → Summary) that preserves task-critical information with probabilistic guarantees.

### Core Objectives

1. **Oracle-Preserving Guarantees**: Implement three consistency conditions (C1: Sufficiency, C2: Idempotence, C3: Merge Consistency) ensuring summaries preserve oracle values f*(d)
2. **Probabilistic Auditing**: Sample-based verification with confidence intervals, avoiding full document re-reading
3. **Bootstrap Optimization**: Use audit violations as training signal for DSPy modules
4. **Production-Ready Engineering**: Parallel execution, rate limiting, caching, human-in-loop oracle routing
5. **Pragmatic OmniThink Integration**: Reuse DSPy scaffolding, tree structures, and UI components where beneficial

### Success Criteria

- **Functional**: Violation rates below ε=0.1 threshold with δ=0.05 confidence on sample documents
- **Performance**: Parallel layer-wise contraction at >10 leaves/sec on 4-worker pool
- **Quality**: Compression ratio 5:1+ while maintaining oracle accuracy
- **Usability**: CLI batch processing + Streamlit tree visualization + human audit interface
- **Deployability**: Docker image, CI tests, PyPI package

### Non-Goals (v0)

- Fine-tuning proprietary models (use prompt optimization only)
- Full RAG/search integration (keep hooks for future)
- Large-scale distributed training infrastructure
- Multi-document cross-referencing

---

## 1. Theoretical Foundation

### 1.1 The Three Consistency Conditions

**C1: Sufficiency (Leaf Level)**
```
∀ leaf nodes i: f*(d_i) = f*(g(d_i))
```
Leaf summarizer g preserves oracle value of raw text d_i.

**C2: Idempotence (Stability)**
```
∀ nodes j: f*(s_j) = f*(g(s_j))
```
Re-summarizing doesn't change oracle value (summary is stable).

**C3: Merge Consistency**
```
∀ internal nodes k with children (L, R):
f*(g(s_L ⊕ s_R)) = f*(s_L ∪ s_R)
```
Merging summaries preserves the oracle of concatenated child summaries.

### 1.2 Probabilistic Audit (Theorem 1)

Sample m = ⌈(2/ε²) ln(2/δ)⌉ nodes uniformly. With probability ≥ 1-δ, at most ε-fraction violate consistency if all sampled nodes pass.

**Key insight**: Audit cost is O(√n log(1/δ)) oracle calls instead of O(n).

### 1.3 Oracle Notation

- **f***: Expensive ground-truth oracle (human annotator or high-grade model)
- **f̂**: Surrogate oracle (DSPy program, cheaper approximation)
- **d_Y**: Distance metric in oracle space Y (exact match, Hamming, L1, etc.)
- **Rubric**: Task definition specifying Y, d_Y, and canonicalization rules

---

## 2. Architecture Overview

### 2.1 Data Flow

```
Document → Chunker → Leaf Nodes (layer=0)
                          ↓
              Parallel LeafSummarize (g)
                          ↓
                  Layer 1 Summaries
                          ↓
              Parallel MergeSummarize (g)
                          ↓
                  Layer 2 Summaries
                          ↓
                        ...
                          ↓
              Root Summary (layer=max_layer)
                          ↓
          ┌───────────────┴───────────────┐
          ↓                               ↓
   AuditEngine (sample)         OracleRouter (f*/f̂)
          ↓                               ↓
   Violations? ──Yes→ BootstrapOptimizer
          ↓                    ↓
         No              Retrain & Rebuild
          ↓
   Export Summary/Tree
```

### 2.2 Layer-Based Tree Architecture

**Key difference from cld**: Use `layer` (height from leaves) instead of `depth` (distance from root) for bottom-up construction.

- **Layer 0**: Leaf nodes with raw text spans
- **Layer k**: Internal nodes at height k above leaves
- **Root**: node.layer = max_layer
- **Virtual depth**: For UI compatibility, `virtual_depth = max_layer - layer` (root has depth 0)

### 2.3 OmniThink Integration Strategy

| Component | Strategy | Notes |
|-----------|----------|-------|
| Node/Tree structures | **Reuse + Extend** | Add `.layer`, `.raw_span`, `.audit_status` |
| DSPy integration | **Reuse** | Keep dspy.Signature patterns, lm wrappers |
| Streamlit UI | **Adapt** | Tree renderer with audit badges, depth mapping |
| Utilities (text processing) | **Reuse** | ArticleTextProcessing, chunking helpers |
| Search/retrieval | **Deprecate** | Remove from core pipeline (optional future) |
| Article generation | **Deprecate** | Expansion-specific, not needed for contraction |

---

## 3. Core Data Structures

### 3.1 Node (Extended from OmniThink)

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

class AuditStatus(Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED_C1 = "failed_c1_sufficiency"
    FAILED_C2 = "failed_c2_idempotence"
    FAILED_C3 = "failed_c3_merge_consistency"

@dataclass
class Node:
    """
    Tree node for hierarchical summarization.
    Extends OmniThink's node structure with layer-based architecture.
    """
    id: str
    layer: int  # Height from leaves (0 = leaf, increases upward)

    # Content
    raw_span: Optional[str] = None  # Only populated for leaves
    summary: Optional[str] = None   # g(raw_span) for leaves, g(child_summaries) for internal

    # Tree structure
    children: List['Node'] = field(default_factory=list)
    parent: Optional['Node'] = None

    # Audit metadata
    audit_status: AuditStatus = AuditStatus.PENDING
    oracle_value: Optional[Any] = None  # Cached f*(node)
    oracle_distance: float = 0.0        # d_Y(f*(raw), f*(summary))

    # Performance tracking
    metadata: Dict[str, Any] = field(default_factory=dict)  # token_count, latency, model_version

    @property
    def is_leaf(self) -> bool:
        return self.layer == 0

    @property
    def virtual_depth(self) -> int:
        """UI-compatible depth (root = 0)."""
        return self._max_layer - self.layer if hasattr(self, '_max_layer') else 0

    def get_text(self) -> str:
        """Get effective text (raw_span for leaves, summary for internal)."""
        return self.raw_span if self.is_leaf else self.summary
```

### 3.2 Tree

```python
@dataclass
class Tree:
    """
    Hierarchical reduction tree built bottom-up from document chunks.
    """
    root: Node
    nodes: Dict[str, Node] = field(default_factory=dict)  # id -> node lookup
    max_layer: int = 0

    # Frontier tracking for iterative construction
    frontier: List[Node] = field(default_factory=list)

    def __post_init__(self):
        self._index_tree(self.root)
        self._compute_max_layer()

    def _index_tree(self, node: Node):
        """Recursively index all nodes."""
        self.nodes[node.id] = node
        for child in node.children:
            self._index_tree(child)

    def _compute_max_layer(self):
        """Set max_layer and propagate to nodes for virtual_depth."""
        self.max_layer = max((n.layer for n in self.nodes.values()), default=0)
        for node in self.nodes.values():
            node._max_layer = self.max_layer

    def get_nodes_at_layer(self, layer: int) -> List[Node]:
        """Get all nodes at specified layer."""
        return [n for n in self.nodes.values() if n.layer == layer]

    def get_leaves(self) -> List[Node]:
        """Get all leaf nodes (layer 0)."""
        return self.get_nodes_at_layer(0)
```

### 3.3 Rubric

```python
from typing import Callable, Any, List, Tuple

@dataclass
class Rubric:
    """
    Oracle task definition: what information to preserve.
    """
    name: str
    description: str  # Task instructions for DSPy modules

    # Oracle space
    output_type: type  # Type of Y (str, List[str], int, Dict, etc.)

    # Oracle functions
    oracle_expensive: Callable[[str], Any]  # f* (ground truth)
    oracle_surrogate: Optional[Callable[[str], Any]] = None  # f̂ (cheap approximation)

    # Distance metric
    distance_function: Callable[[Any, Any], Tuple[bool, float]]  # (matches, distance)

    # Canonicalization (for idempotence)
    canonicalizer: Optional[Callable[[str], str]] = None

    # DSPy prompt components
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def extract(self, text: str, use_expensive: bool = False) -> Any:
        """Extract oracle value from text."""
        if use_expensive:
            return self.oracle_expensive(text)
        elif self.oracle_surrogate:
            return self.oracle_surrogate(text)
        else:
            return self.oracle_expensive(text)

    def compare(self, val1: Any, val2: Any) -> Tuple[bool, float]:
        """Compare two oracle values. Returns (matches, distance)."""
        return self.distance_function(val1, val2)
```

### 3.4 AuditRecord

```python
@dataclass
class AuditRecord:
    """Record of a single audit check."""
    node_id: str
    check_type: str  # "c1_sufficiency", "c2_idempotence", "c3_merge_consistency"
    oracle_raw: Any
    oracle_summary: Any
    matches: bool
    distance: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)  # prompt version, model, etc.
```

### 3.5 Exemplar (for optimization)

```python
@dataclass
class Exemplar:
    """Training example from audit violation."""
    rubric_name: str
    input_text: str  # raw_span or concatenated child summaries
    generated_summary: str
    expected_oracle: Any
    actual_oracle: Any
    violation_type: str  # "c1", "c2", "c3"
    node_id: str
```

---

## 4. DSPy Modules

### 4.1 LeafSummarizer

```python
import dspy
from typing import Union

class LeafSummarize(dspy.Signature):
    """
    Compress raw text block into summary preserving oracle information.
    Enforces C1: f*(raw_text) = f*(summary)
    """
    rubric = dspy.InputField(
        prefix="Task definition (information to preserve):\n",
        format=str
    )
    raw_text = dspy.InputField(
        prefix="Raw text block:\n",
        format=str
    )
    summary = dspy.OutputField(
        prefix="Compressed summary (preserve rubric info, ~30% original length):\n",
        format=str
    )

class LeafSummarizer(dspy.Module):
    def __init__(self, lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.lm = lm
        self.predictor = dspy.ChainOfThought(LeafSummarize)

    def forward(self, rubric_desc: str, raw_text: str) -> dspy.Prediction:
        with dspy.settings.context(lm=self.lm):
            return self.predictor(rubric=rubric_desc, raw_text=raw_text)
```

### 4.2 MergeSummarizer

```python
class MergeSummarize(dspy.Signature):
    """
    Merge two child summaries into parent summary.
    Enforces C3: f*(merged) = f*(left ⊕ right)
    """
    rubric = dspy.InputField(prefix="Task definition:\n", format=str)
    left_summary = dspy.InputField(prefix="Left child summary:\n", format=str)
    right_summary = dspy.InputField(prefix="Right child summary:\n", format=str)
    merged_summary = dspy.OutputField(
        prefix="Merged parent summary (preserve rubric info from both children):\n",
        format=str
    )

class MergeSummarizer(dspy.Module):
    def __init__(self, lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.lm = lm
        self.predictor = dspy.ChainOfThought(MergeSummarize)

    def forward(self, rubric_desc: str, left: str, right: str) -> dspy.Prediction:
        with dspy.settings.context(lm=self.lm):
            return self.predictor(rubric=rubric_desc, left_summary=left, right_summary=right)
```

### 4.3 OracleApproximator (Surrogate f̂)

```python
class OracleApproximate(dspy.Signature):
    """Surrogate oracle f̂ for cheap approximation of f*."""
    rubric = dspy.InputField(prefix="Oracle task definition:\n", format=str)
    text = dspy.InputField(prefix="Text to analyze:\n", format=str)
    oracle_value = dspy.OutputField(prefix="Extracted oracle value:\n", format=str)

class OracleApproximator(dspy.Module):
    def __init__(self, lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.lm = lm
        self.predictor = dspy.ChainOfThought(OracleApproximate)

    def forward(self, rubric_desc: str, text: str) -> Any:
        with dspy.settings.context(lm=self.lm):
            result = self.predictor(rubric=rubric_desc, text=text)
            return result.oracle_value
```

---

## 5. Oracle System with Human-in-Loop

### 5.1 OracleRouter

```python
import logging
from typing import Dict, Any

class OracleRouter:
    """
    Routes oracle queries to f* or f̂ based on policy.
    Supports human escalation for flagged/uncertain nodes.
    """
    def __init__(self, rubric: Rubric, use_surrogate_default: bool = True):
        self.rubric = rubric
        self.use_surrogate = use_surrogate_default
        self.cache: Dict[str, Any] = {}  # text_hash -> oracle_value
        self.stats = {
            'expensive_calls': 0,
            'surrogate_calls': 0,
            'cache_hits': 0,
            'human_escalations': 0
        }
        self.logger = logging.getLogger(__name__)

    def __call__(self, text: str, force_expensive: bool = False,
                 uncertainty_threshold: float = 0.9) -> Any:
        """
        Extract oracle value with routing logic.

        Args:
            text: Text to extract oracle from
            force_expensive: Always use f* (for audit ground truth)
            uncertainty_threshold: Escalate to f* if f̂ confidence < threshold
        """
        cache_key = hash((text, self.rubric.name))

        # Check cache
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]

        # Routing logic
        if force_expensive:
            value = self._call_expensive(text)
        elif self.use_surrogate:
            value, confidence = self._call_surrogate_with_confidence(text)
            if confidence < uncertainty_threshold:
                self.logger.info(f"Escalating to f* (confidence {confidence:.2f} < {uncertainty_threshold})")
                value = self._call_expensive(text)
        else:
            value = self._call_expensive(text)

        self.cache[cache_key] = value
        return value

    def _call_expensive(self, text: str) -> Any:
        self.stats['expensive_calls'] += 1
        return self.rubric.extract(text, use_expensive=True)

    def _call_surrogate_with_confidence(self, text: str) -> tuple[Any, float]:
        self.stats['surrogate_calls'] += 1
        value = self.rubric.extract(text, use_expensive=False)
        # TODO: Extract confidence from DSPy prediction metadata
        confidence = 1.0  # Placeholder
        return value, confidence

    def request_human_annotation(self, node_id: str, text: str) -> Any:
        """Escalate to human annotator (stores in queue for batch processing)."""
        self.stats['human_escalations'] += 1
        # TODO: Write to human annotation queue
        self.logger.warning(f"Human annotation requested for node {node_id}")
        return None
```

---

## 6. Tree Construction with Parallelization

### 6.1 DocumentChunker

```python
from typing import List, Tuple

class DocumentChunker:
    """Split document into leaf-sized chunks with optional overlap."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: str) -> List[Tuple[str, int, int]]:
        """
        Returns list of (chunk_text, start_offset, end_offset).
        Ensures power-of-2 chunks for balanced binary tree.
        """
        words = document.split()
        chunks = []

        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            start = i
            end = min(i + self.chunk_size, len(words))
            chunks.append((chunk_text, start, end))
            i += (self.chunk_size - self.overlap)

        # Pad to power of 2
        target_count = 2 ** math.ceil(math.log2(len(chunks)))
        while len(chunks) < target_count:
            # Split largest chunk
            largest_idx = max(range(len(chunks)), key=lambda j: len(chunks[j][0]))
            text, start, end = chunks[largest_idx]
            mid = (end - start) // 2
            chunks[largest_idx] = (text[:mid], start, start + mid)
            chunks.insert(largest_idx + 1, (text[mid:], start + mid, end))

        return chunks[:target_count]
```

### 6.2 TreeBuilder with Parallel Execution

```python
import concurrent.futures
from typing import List
import logging

class TreeBuilder:
    """Construct tree bottom-up with parallel layer processing."""

    def __init__(self,
                 leaf_summarizer: LeafSummarizer,
                 merge_summarizer: MergeSummarizer,
                 rubric: Rubric,
                 max_workers: int = 4,
                 batch_size: int = 10):
        self.leaf_summarizer = leaf_summarizer
        self.merge_summarizer = merge_summarizer
        self.rubric = rubric
        self.max_workers = max_workers
        self.batch_size = batch_size  # For rate limiting
        self.logger = logging.getLogger(__name__)

    def build_tree(self, chunks: List[Tuple[str, int, int]]) -> Tree:
        """Build complete tree from chunks with parallel execution."""

        # Create leaf nodes
        leaf_nodes = self._create_leaf_layer(chunks)

        # Iteratively merge layers
        current_layer = leaf_nodes
        layer_num = 1

        while len(current_layer) > 1:
            self.logger.info(f"Processing layer {layer_num}: {len(current_layer)} nodes")
            next_layer = self._contract_layer_parallel(current_layer, layer_num)
            current_layer = next_layer
            layer_num += 1

        root = current_layer[0]
        return Tree(root=root)

    def _create_leaf_layer(self, chunks: List[Tuple[str, int, int]]) -> List[Node]:
        """Create and summarize leaf nodes in parallel."""
        leaf_nodes = [
            Node(id=f"leaf_{i}", layer=0, raw_span=text)
            for i, (text, start, end) in enumerate(chunks)
        ]

        # Parallel leaf summarization with batching
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for node in leaf_nodes:
                future = executor.submit(
                    self._summarize_leaf, node, self.rubric.description
                )
                futures.append((future, node))

                # Rate limiting: wait if batch full
                if len(futures) >= self.batch_size:
                    for f, n in futures:
                        result = f.result()
                        n.summary = result.summary
                        n.metadata['leaf_tokens'] = len(result.summary.split())
                    futures = []

            # Process remaining
            for f, n in futures:
                result = f.result()
                n.summary = result.summary
                n.metadata['leaf_tokens'] = len(result.summary.split())

        return leaf_nodes

    def _summarize_leaf(self, node: Node, rubric_desc: str) -> dspy.Prediction:
        """Summarize single leaf node."""
        return self.leaf_summarizer(rubric_desc, node.raw_span)

    def _contract_layer_parallel(self, nodes: List[Node], layer_num: int) -> List[Node]:
        """Merge pairs of nodes into next layer with parallel execution."""
        next_layer = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i + 1] if i + 1 < len(nodes) else None

                if right is None:
                    # Odd node, promote to next layer
                    next_layer.append(left)
                    continue

                parent = Node(
                    id=f"node_L{layer_num}_{i//2}",
                    layer=layer_num,
                    children=[left, right]
                )
                left.parent = parent
                right.parent = parent

                future = executor.submit(
                    self._merge_nodes, parent, left, right, self.rubric.description
                )
                futures.append((future, parent))

            # Collect results
            for f, parent in futures:
                result = f.result()
                parent.summary = result.merged_summary
                parent.metadata['merge_tokens'] = len(result.merged_summary.split())
                next_layer.append(parent)

        return next_layer

    def _merge_nodes(self, parent: Node, left: Node, right: Node,
                     rubric_desc: str) -> dspy.Prediction:
        """Merge two child nodes."""
        return self.merge_summarizer(rubric_desc, left.summary, right.summary)
```

---

## 7. Audit Engine with Confidence Intervals

### 7.1 AuditEngine Implementation

```python
import random
import math
from typing import List, Tuple
from scipy import stats  # For Wilson score interval

class AuditEngine:
    """Probabilistic audit with statistical confidence intervals."""

    def __init__(self, oracle_router: OracleRouter, rubric: Rubric):
        self.oracle = oracle_router
        self.rubric = rubric
        self.logger = logging.getLogger(__name__)

    def compute_sample_size(self, epsilon: float, delta: float) -> int:
        """Theorem 1: m = ⌈(2/ε²) ln(2/δ)⌉"""
        return math.ceil((2 / (epsilon ** 2)) * math.log(2 / delta))

    def audit(self, tree: Tree, epsilon: float = 0.1, delta: float = 0.05) -> Dict[str, Any]:
        """
        Run complete audit with all three consistency checks.

        Returns:
            {
                'sample_size': int,
                'violations': List[AuditRecord],
                'violation_rate': float,
                'confidence_interval': (lower, upper),
                'passed': bool
            }
        """
        sample_size = self.compute_sample_size(epsilon, delta)
        all_nodes = list(tree.nodes.values())
        sampled_nodes = random.sample(all_nodes, min(sample_size, len(all_nodes)))

        violations = []

        for node in sampled_nodes:
            # Check C1 for leaves
            if node.is_leaf:
                record = self._check_c1_sufficiency(node)
                if not record.matches:
                    violations.append(record)
                    node.audit_status = AuditStatus.FAILED_C1

            # Check C3 for internal nodes
            if not node.is_leaf and len(node.children) == 2:
                record = self._check_c3_merge_consistency(node)
                if not record.matches:
                    violations.append(record)
                    node.audit_status = AuditStatus.FAILED_C3

            # Check C2 for all nodes (sample subset for cost)
            if random.random() < 0.3:  # Sample 30% for idempotence
                record = self._check_c2_idempotence(node)
                if not record.matches:
                    violations.append(record)
                    node.audit_status = AuditStatus.FAILED_C2

            if node.audit_status == AuditStatus.PENDING:
                node.audit_status = AuditStatus.PASSED

        # Compute statistics
        violation_rate = len(violations) / sample_size
        ci_lower, ci_upper = self._wilson_score_interval(
            len(violations), sample_size, 1 - delta
        )

        passed = ci_upper < epsilon

        self.logger.info(
            f"Audit: {len(violations)}/{sample_size} violations "
            f"(rate={violation_rate:.3f}, CI=[{ci_lower:.3f}, {ci_upper:.3f}])"
        )

        return {
            'sample_size': sample_size,
            'violations': violations,
            'violation_rate': violation_rate,
            'confidence_interval': (ci_lower, ci_upper),
            'passed': passed,
            'epsilon': epsilon,
            'delta': delta
        }

    def _check_c1_sufficiency(self, node: Node) -> AuditRecord:
        """C1: f*(raw_text) = f*(summary)"""
        oracle_raw = self.oracle(node.raw_span, force_expensive=True)
        oracle_summary = self.oracle(node.summary, force_expensive=True)
        matches, distance = self.rubric.compare(oracle_raw, oracle_summary)

        return AuditRecord(
            node_id=node.id,
            check_type="c1_sufficiency",
            oracle_raw=oracle_raw,
            oracle_summary=oracle_summary,
            matches=matches,
            distance=distance,
            timestamp=time.time()
        )

    def _check_c3_merge_consistency(self, node: Node) -> AuditRecord:
        """C3: f*(merged) = f*(concat_children)"""
        left, right = node.children[0], node.children[1]
        concat_text = left.summary + " " + right.summary

        oracle_merged = self.oracle(node.summary, force_expensive=True)
        oracle_concat = self.oracle(concat_text, force_expensive=True)
        matches, distance = self.rubric.compare(oracle_merged, oracle_concat)

        return AuditRecord(
            node_id=node.id,
            check_type="c3_merge_consistency",
            oracle_raw=oracle_concat,
            oracle_summary=oracle_merged,
            matches=matches,
            distance=distance,
            timestamp=time.time()
        )

    def _check_c2_idempotence(self, node: Node) -> AuditRecord:
        """C2: f*(summary) = f*(g(summary))"""
        oracle_original = self.oracle(node.summary, force_expensive=True)

        # Re-summarize
        if node.is_leaf:
            re_summary = self.leaf_summarizer(self.rubric.description, node.summary).summary
        else:
            # Re-merge from children
            left, right = node.children[0], node.children[1]
            re_summary = self.merge_summarizer(
                self.rubric.description, left.summary, right.summary
            ).merged_summary

        oracle_resummary = self.oracle(re_summary, force_expensive=True)
        matches, distance = self.rubric.compare(oracle_original, oracle_resummary)

        return AuditRecord(
            node_id=node.id,
            check_type="c2_idempotence",
            oracle_raw=oracle_original,
            oracle_summary=oracle_resummary,
            matches=matches,
            distance=distance,
            timestamp=time.time()
        )

    def _wilson_score_interval(self, successes: int, trials: int,
                               confidence: float) -> Tuple[float, float]:
        """Compute Wilson score confidence interval for binomial proportion."""
        if trials == 0:
            return (0.0, 1.0)

        p_hat = successes / trials
        z = stats.norm.ppf((1 + confidence) / 2)

        denominator = 1 + z**2 / trials
        center = (p_hat + z**2 / (2 * trials)) / denominator
        margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials) / denominator

        return (max(0, center - margin), min(1, center + margin))
```

---

## 8. Bootstrap Optimization Loop

```python
from typing import List
import dspy.teleprompt

class BootstrapOptimizer:
    """Optimize DSPy modules using audit violations as training signal."""

    def __init__(self, tree_builder: TreeBuilder, audit_engine: AuditEngine,
                 max_iterations: int = 5):
        self.tree_builder = tree_builder
        self.audit_engine = audit_engine
        self.max_iterations = max_iterations
        self.exemplar_bank: List[Exemplar] = []

    def optimize_loop(self, document: str, chunks: List[Tuple[str, int, int]],
                     epsilon: float = 0.1, delta: float = 0.05) -> Tree:
        """
        Bootstrap optimization loop.

        1. Build tree
        2. Audit
        3. If violations, create exemplars and optimize
        4. Repeat until pass or max iterations
        """
        for iteration in range(self.max_iterations):
            print(f"\n=== Bootstrap Iteration {iteration + 1}/{self.max_iterations} ===")

            # Build tree
            tree = self.tree_builder.build_tree(chunks)

            # Audit
            audit_result = self.audit_engine.audit(tree, epsilon, delta)
            print(f"Violations: {len(audit_result['violations'])}/{audit_result['sample_size']}")
            print(f"Confidence interval: {audit_result['confidence_interval']}")

            if audit_result['passed']:
                print("✓ Audit passed!")
                return tree

            # Create exemplars from violations
            exemplars = self._create_exemplars(audit_result['violations'], tree)
            self.exemplar_bank.extend(exemplars)

            # Optimize DSPy modules
            self._optimize_modules(exemplars)

        print(f"⚠ Did not converge after {self.max_iterations} iterations")
        return tree

    def _create_exemplars(self, violations: List[AuditRecord], tree: Tree) -> List[Exemplar]:
        """Convert violations to training examples."""
        exemplars = []

        for record in violations:
            node = tree.nodes[record.node_id]

            if record.check_type == "c1_sufficiency":
                exemplars.append(Exemplar(
                    rubric_name=self.audit_engine.rubric.name,
                    input_text=node.raw_span,
                    generated_summary=node.summary,
                    expected_oracle=record.oracle_raw,
                    actual_oracle=record.oracle_summary,
                    violation_type="c1",
                    node_id=node.id
                ))

            elif record.check_type == "c3_merge_consistency":
                left, right = node.children[0], node.children[1]
                concat = left.summary + " " + right.summary
                exemplars.append(Exemplar(
                    rubric_name=self.audit_engine.rubric.name,
                    input_text=concat,
                    generated_summary=node.summary,
                    expected_oracle=record.oracle_raw,
                    actual_oracle=record.oracle_summary,
                    violation_type="c3",
                    node_id=node.id
                ))

        return exemplars

    def _optimize_modules(self, exemplars: List[Exemplar]):
        """Use DSPy teleprompter to optimize summarizers."""

        # Separate by violation type
        c1_exemplars = [e for e in exemplars if e.violation_type == "c1"]
        c3_exemplars = [e for e in exemplars if e.violation_type == "c3"]

        # Convert to DSPy Examples
        def to_dspy_example(ex: Exemplar) -> dspy.Example:
            return dspy.Example(
                rubric=self.audit_engine.rubric.description,
                input_text=ex.input_text,
                expected_oracle=str(ex.expected_oracle)
            ).with_inputs("rubric", "input_text")

        # Optimize LeafSummarizer
        if c1_exemplars:
            leaf_trainset = [to_dspy_example(e) for e in c1_exemplars]
            teleprompter = dspy.teleprompt.BootstrapFewShot(
                metric=self._oracle_preservation_metric,
                max_bootstrapped_demos=4
            )
            self.tree_builder.leaf_summarizer = teleprompter.compile(
                self.tree_builder.leaf_summarizer,
                trainset=leaf_trainset
            )

        # Optimize MergeSummarizer
        if c3_exemplars:
            merge_trainset = [to_dspy_example(e) for e in c3_exemplars]
            teleprompter = dspy.teleprompt.BootstrapFewShot(
                metric=self._oracle_preservation_metric,
                max_bootstrapped_demos=4
            )
            self.tree_builder.merge_summarizer = teleprompter.compile(
                self.tree_builder.merge_summarizer,
                trainset=merge_trainset
            )

    def _oracle_preservation_metric(self, example, pred, trace=None) -> bool:
        """Metric: does generated summary preserve oracle?"""
        oracle_pred = self.audit_engine.oracle(pred.summary if hasattr(pred, 'summary') else pred.merged_summary)
        oracle_expected = example.expected_oracle
        matches, _ = self.audit_engine.rubric.compare(oracle_pred, oracle_expected)
        return matches
```

---

## 9. CLI and UI

### 9.1 CLI Commands

```bash
# Ingest document and build tree
thinkingtrees ingest --doc article.txt --rubric sentiment --output tree.json

# Run audit
thinkingtrees audit --tree tree.json --epsilon 0.1 --delta 0.05 --report audit_report.json

# Bootstrap optimization
thinkingtrees optimize --doc article.txt --rubric sentiment --iterations 5 --output optimized_tree.json

# Export summary
thinkingtrees export --tree tree.json --format markdown --output summary.md
```

### 9.2 Streamlit UI

Reuse OmniThink's Streamlit shell with adaptations:

- **Tree Viewer**: Hierarchical visualization with layer-based rendering (use `virtual_depth` for compatibility)
- **Audit Badges**: Color-code nodes by `audit_status` (green=passed, red=failed, gray=pending)
- **Node Inspector**: Show raw_span (leaves), summary, oracle values, violation details
- **Human Annotation**: Edit box for correcting oracle values, saves to exemplar bank
- **Controls**: Run audit, re-optimize, export reports

---

## 10. Implementation Milestones

### Milestone 1: Scaffold & Core Structures (Week 1)
- [ ] Repo setup with package layout (`thinkingtrees/`)
- [ ] Data structures: `Node`, `Tree`, `Rubric`, `AuditRecord`, `Exemplar`
- [ ] Unit tests for data structures
- [ ] Configuration system (YAML/JSON)

### Milestone 2: Chunking & Tree Construction (Week 2)
- [ ] `DocumentChunker` with power-of-2 balancing
- [ ] `TreeBuilder` with serial layer contraction
- [ ] Serialize/deserialize tree to JSON
- [ ] Integration test: chunked doc → tree

### Milestone 3: DSPy Modules (Week 2-3)
- [ ] `LeafSummarizer` and `MergeSummarizer` signatures
- [ ] `OracleApproximator` for f̂
- [ ] Test on sample documents with simple rubrics
- [ ] LM provider configuration (OpenAI, local models)

### Milestone 4: Parallel Execution (Week 3)
- [ ] ThreadPoolExecutor integration in `TreeBuilder`
- [ ] Rate limiting with batch_size parameter
- [ ] Performance benchmark: >10 leaves/sec on 4 workers
- [ ] Structured logging for progress tracking

### Milestone 5: Oracle Routing (Week 3-4)
- [ ] `OracleRouter` with cache and statistics
- [ ] Human-in-loop escalation hooks
- [ ] Example rubrics: classification, entity extraction, counting
- [ ] Unit tests for distance metrics

### Milestone 6: Audit Engine (Week 4-5)
- [ ] `AuditEngine` with C1, C2, C3 checks
- [ ] Sample size calculation (Theorem 1)
- [ ] Wilson score confidence intervals
- [ ] Audit report generation (JSON/CSV)
- [ ] Test with planted violations

### Milestone 7: Bootstrap Optimization (Week 5-6)
- [ ] `BootstrapOptimizer` with exemplar creation
- [ ] DSPy teleprompter integration (BootstrapFewShot/MIPRO)
- [ ] Oracle preservation metric
- [ ] Test convergence on sample documents

### Milestone 8: CLI & Deployment (Week 6-7)
- [ ] CLI commands: ingest, audit, optimize, export
- [ ] Streamlit UI with tree viewer and audit overlays
- [ ] Docker image with CPU baseline
- [ ] PyPI package setup (setup.py, requirements.txt)

### Milestone 9: Testing & Hardening (Week 7-8)
- [ ] Unit test suite (80%+ coverage)
- [ ] Integration tests with fixtures
- [ ] Performance benchmarks on synthetic large docs
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Documentation (README, API reference, examples)

---

## 11. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **Oracle cost/latency** | Cache aggressively, use f̂ by default, batch oracle calls, human-in-loop for high-value nodes |
| **Context limits at high layers** | Rely on inductive C3 checks (audit children, trust transitivity), optional truncation for full concat checks |
| **Rate limits under parallelism** | Tune `batch_size` and `max_workers`, add semaphore-based dispatcher, exponential backoff on 429 errors |
| **Task schema ambiguity** | Build rubric library with canonicalization examples, enforce strict output types, validate in tests |
| **Audit statistical validity** | Use proper confidence intervals (Wilson score), document sample size assumptions, add sensitivity analysis |
| **DSPy optimization doesn't converge** | Set max_iterations cap, fallback to manual prompt engineering, collect human exemplars for hard cases |
| **UI complexity blocks releases** | Keep UI minimal (tree + audit badges only), ensure CLI+JSON exports work standalone, defer advanced features |

---

## 12. Success Metrics

### Functional Quality
- **C1 violation rate** < 10% on leaves (ε=0.1, δ=0.05)
- **C3 violation rate** < 10% on internal nodes
- **C2 violation rate** < 5% (idempotence easier to achieve)

### Performance
- **Leaf summarization throughput**: >10 leaves/sec (4 workers, gpt-4o-mini)
- **Tree construction time**: <2 min for 64-leaf document
- **Audit time**: <1 min for sample_size=100

### Compression
- **Compression ratio**: 5:1 to 10:1 (document:summary length)
- **Oracle preservation**: >90% accuracy vs ground truth f*

### Engineering
- **Test coverage**: >80% line coverage
- **CI green**: All tests pass on main branch
- **Package health**: Installable via pip, Docker image builds successfully

---

## 13. Future Extensions (Post-v0)

- **Multi-way trees**: k-ary trees with adaptive branching factor
- **Adaptive chunking**: Semantic segmentation (paragraph/section boundaries)
- **Incremental updates**: Edit detection and localized tree rebuilds
- **Multi-document**: Cross-document merge with entity resolution
- **Learned oracles**: End-to-end neural oracle training
- **Oracle discovery**: Auto-infer rubrics from document collections
- **Active learning**: Human-in-loop for maximal information gain samples
- **Distributed execution**: Ray/Dask for large-scale corpora

---

## Conclusion

This v2 implementation plan synthesizes:

1. **Theoretical rigor** (cld): Mathematical foundations, DSPy module designs, detailed data structures
2. **Production engineering** (cdx): Parallelization, rate limiting, oracle routing, risk mitigation

The result is a **comprehensive yet pragmatic roadmap** for building ThinkingTrees as a production-ready OPS system. The 8-week milestone plan provides clear deliverables, while the risk mitigation and success criteria ensure quality gates are met before each release.

**Next immediate steps**:
1. Create repo scaffold with package layout
2. Implement core data structures (Node, Tree, Rubric)
3. Write unit tests for data structure invariants
4. Set up CI pipeline with pytest
5. Begin Milestone 1 (Scaffold & Core Structures)
