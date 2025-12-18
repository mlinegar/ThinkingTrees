# ThinkingTrees Implementation Plan

## Executive Summary

**ThinkingTrees** is a hierarchical document summarization framework implementing the Oracle-Preserving Summarization (OPS) methodology from the research paper "Hierarchical Summarization with Oracle-Preserving Guarantees". This system inverts the OmniThink architecture: while OmniThink is an **expansion engine** (topic → MindMap → Article), ThinkingTrees is a **reduction engine** (Document → ReductionTree → Summary).

### Core Objectives

1. **Oracle-Preserving Guarantees**: Implement the three consistency conditions (C1: Sufficiency, C2: Idempotence, C3: Merge Consistency)
2. **Probabilistic Auditing**: Enable efficient verification without full document re-reading
3. **Bootstrap Optimization**: Improve summarizers using audit violations as training signal
4. **DSPy Integration**: Leverage the existing DSPy framework from OmniThink
5. **Modular Architecture**: Build reusable components compatible with OmniThink's structure

### Key Innovation

ThinkingTrees ensures that summaries preserve critical information defined by an **oracle function** (f*). The oracle maps documents to task-specific values (labels, entity lists, numerical aggregates, etc.). The three consistency conditions provide mathematical guarantees that the hierarchical summarization process maintains oracle accuracy across all tree levels.

---

## 1. Theoretical Foundation

### 1.1 The Three Consistency Conditions

From the OPS paper, we implement:

**C1: Sufficiency (Leaf Level)**
```
For all leaf nodes i: f*(d_i) = f*(s_i)
```
The leaf summary preserves the oracle value of the raw text.

**C2: Idempotence (Stability)**
```
For all nodes j: f*(s_j) = f*(σ(s_j))
```
Re-summarizing a summary doesn't change its oracle value.

**C3: Merge Consistency**
```
For all internal nodes k with children (L, R):
f*(σ(s_L, s_R)) = f*(s_L ∪ s_R)
```
Merging summaries then extracting oracle equals extracting from concatenated summaries.

### 1.2 Probabilistic Audit

**Theorem 1** (Section 4 of paper): With probability ≥ 1-δ, at most ε-fraction of nodes violate consistency if:
- Sample m = ⌈(2/ε²) ln(2/δ)⌉ nodes uniformly
- Check each for violations
- Accept if all sampled nodes pass

**Theorem 2**: The audit can be performed with O(√n log(1/δ)) oracle calls instead of O(n).

### 1.3 Bootstrap Training Loop

1. Build initial tree with weak summarizers
2. Run probabilistic audit
3. Collect violation nodes as negative examples
4. Use violations to optimize DSPy modules
5. Rebuild tree with improved summarizers
6. Repeat until audit passes

---

## 2. Architecture Overview

### 2.1 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ThinkingTrees Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Document → Chunker → ReductionTree Builder                │
│                            ↓                                │
│                    ┌──────────────┐                         │
│                    │  TreeNode[]  │                         │
│                    └──────┬───────┘                         │
│                           ↓                                 │
│              ┌────────────────────────┐                     │
│              │  LeafSummarizer (DSPy) │                     │
│              └────────────┬───────────┘                     │
│                           ↓                                 │
│              ┌────────────────────────┐                     │
│              │ MergeSummarizer (DSPy) │                     │
│              └────────────┬───────────┘                     │
│                           ↓                                 │
│              ┌────────────────────────┐                     │
│              │   OracleExtractor      │                     │
│              └────────────┬───────────┘                     │
│                           ↓                                 │
│              ┌────────────────────────┐                     │
│              │     AuditEngine        │                     │
│              └────────────┬───────────┘                     │
│                           ↓                                 │
│         Pass? ────No────→ BootstrapOptimizer               │
│           │                      ↓                          │
│          Yes              Re-train & Rebuild                │
│           ↓                                                 │
│    Final Summary + Tree                                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Mapping OPS Concepts to OmniThink Structures

| OPS Concept | OmniThink Analog | ThinkingTrees Implementation |
|-------------|------------------|------------------------------|
| Document chunks | MindPoint.info snippets | TreeNode.raw_text_span |
| Leaf summary | MindPoint.concept | TreeNode.summary (leaf) |
| Merge summary | - | TreeNode.summary (internal) |
| Oracle f* | - | Oracle class with __call__ |
| Rubric Y | - | Rubric dataclass |
| Binary tree | MindMap tree | ReductionTree |
| Audit | - | AuditEngine |
| Bootstrap | - | ConsistencyBootstrap |

---

## 3. Core Data Structures

### 3.1 TreeNode

```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class AuditStatus(Enum):
    PENDING = "pending"
    PASSED = "passed"
    VIOLATED_C1 = "violated_c1"
    VIOLATED_C2 = "violated_c2"
    VIOLATED_C3 = "violated_c3"

class ViolationType(Enum):
    C1_SUFFICIENCY = "c1"
    C2_IDEMPOTENCE = "c2"
    C3_MERGE_CONSISTENCY = "c3"

@dataclass
class TreeNode:
    """Hierarchical node in the reduction tree."""
    node_id: str
    is_leaf: bool
    depth: int

    # Content
    raw_text_span: Optional[str] = None  # Only for leaves
    summary: Optional[str] = None

    # Tree structure
    parent: Optional['TreeNode'] = None
    left_child: Optional['TreeNode'] = None
    right_child: Optional['TreeNode'] = None

    # Audit metadata
    audit_status: AuditStatus = AuditStatus.PENDING
    violation_type: Optional[ViolationType] = None
    oracle_value: Optional[any] = None  # Cached f*(node)

    # DSPy chain-of-thought
    reasoning: Optional[str] = None

    def is_root(self) -> bool:
        return self.parent is None

    def get_siblings(self) -> Optional['TreeNode']:
        if not self.parent:
            return None
        return self.parent.right_child if self == self.parent.left_child else self.parent.left_child

    def get_text(self) -> str:
        """Get the effective text representation of this node."""
        return self.raw_text_span if self.is_leaf else self.summary
```

### 3.2 ReductionTree

```python
from typing import List, Dict
import networkx as nx

class ReductionTree:
    """Binary tree structure for hierarchical document reduction."""

    def __init__(self, root: TreeNode):
        self.root = root
        self.nodes: Dict[str, TreeNode] = {}
        self._index_tree(root)

    def _index_tree(self, node: TreeNode):
        """Recursively index all nodes by ID."""
        self.nodes[node.node_id] = node
        if node.left_child:
            self._index_tree(node.left_child)
        if node.right_child:
            self._index_tree(node.right_child)

    def get_leaves(self) -> List[TreeNode]:
        """Return all leaf nodes in left-to-right order."""
        leaves = []
        def traverse(node):
            if node.is_leaf:
                leaves.append(node)
            else:
                if node.left_child:
                    traverse(node.left_child)
                if node.right_child:
                    traverse(node.right_child)
        traverse(self.root)
        return leaves

    def get_nodes_at_depth(self, depth: int) -> List[TreeNode]:
        """Return all nodes at a specific depth."""
        nodes = []
        def traverse(node, current_depth):
            if current_depth == depth:
                nodes.append(node)
            elif current_depth < depth:
                if node.left_child:
                    traverse(node.left_child, current_depth + 1)
                if node.right_child:
                    traverse(node.right_child, current_depth + 1)
        traverse(self.root, 0)
        return nodes

    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph for visualization."""
        G = nx.DiGraph()
        def add_edges(node):
            if node.left_child:
                G.add_edge(node.node_id, node.left_child.node_id)
                add_edges(node.left_child)
            if node.right_child:
                G.add_edge(node.node_id, node.right_child.node_id)
                add_edges(node.right_child)
        add_edges(self.root)
        return G

    def get_summary(self) -> str:
        """Return the root summary (final compressed document)."""
        return self.root.summary
```

### 3.3 Rubric (Oracle Definition)

```python
from typing import Any, Callable, Dict
from dataclasses import dataclass

@dataclass
class Rubric:
    """
    Defines the oracle space Y and extraction logic.

    Examples:
    - Classification: Y = {positive, negative, neutral}
    - Entity extraction: Y = List[str] (entity names)
    - Counting: Y = int (number of events)
    - Tuple extraction: Y = List[Tuple] (structured facts)
    """
    name: str
    description: str
    output_type: type  # The type of Y
    oracle_function: Callable[[str], Any]  # The expensive f*(d)
    comparison_function: Callable[[Any, Any], bool]  # Equality check in Y

    # Optional: surrogate oracle for cheaper approximation
    surrogate_function: Optional[Callable[[str], Any]] = None

    # Prompt components for DSPy modules
    task_description: str = ""
    examples: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = []

    def extract_oracle(self, text: str, use_surrogate: bool = False) -> Any:
        """Extract oracle value from text."""
        if use_surrogate and self.surrogate_function:
            return self.surrogate_function(text)
        return self.oracle_function(text)

    def values_match(self, val1: Any, val2: Any) -> bool:
        """Check if two oracle values are equivalent."""
        return self.comparison_function(val1, val2)
```

---

## 4. DSPy Modules

### 4.1 LeafSummarizer

```python
import dspy
from typing import Union

class LeafSummarize(dspy.Signature):
    """
    Compress a raw text block into a summary that preserves all information
    needed to answer the rubric's oracle function.

    Constraint: f*(raw_text) must equal f*(summary) (C1: Sufficiency)
    """
    rubric = dspy.InputField(
        prefix="Oracle definition (what information to preserve):\n",
        format=str
    )
    raw_text = dspy.InputField(
        prefix="Raw text block to summarize:\n",
        format=str
    )
    summary = dspy.OutputField(
        prefix="Compressed summary preserving rubric information:\n",
        format=str
    )

class LeafSummarizer(dspy.Module):
    """DSPy module for leaf-level summarization."""

    def __init__(self, lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.lm = lm
        self.predictor = dspy.ChainOfThought(LeafSummarize)

    def forward(self, rubric_desc: str, raw_text: str) -> dspy.Prediction:
        with dspy.settings.context(lm=self.lm):
            result = self.predictor(
                rubric=rubric_desc,
                raw_text=raw_text
            )
        return result
```

### 4.2 MergeSummarizer

```python
class MergeSummarize(dspy.Signature):
    """
    Merge two child summaries into a parent summary that preserves oracle information.

    Constraints:
    - f*(merged_summary) = f*(left_summary ∪ right_summary) (C3: Merge Consistency)
    - f*(merged_summary) = f*(σ(merged_summary)) (C2: Idempotence)
    """
    rubric = dspy.InputField(
        prefix="Oracle definition:\n",
        format=str
    )
    left_summary = dspy.InputField(
        prefix="Left child summary:\n",
        format=str
    )
    right_summary = dspy.InputField(
        prefix="Right child summary:\n",
        format=str
    )
    merged_summary = dspy.OutputField(
        prefix="Merged parent summary:\n",
        format=str
    )

class MergeSummarizer(dspy.Module):
    """DSPy module for merging two summaries."""

    def __init__(self, lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.lm = lm
        self.predictor = dspy.ChainOfThought(MergeSummarize)

    def forward(self, rubric_desc: str, left: str, right: str) -> dspy.Prediction:
        with dspy.settings.context(lm=self.lm):
            result = self.predictor(
                rubric=rubric_desc,
                left_summary=left,
                right_summary=right
            )
        return result
```

### 4.3 OracleExtractor

```python
class ExtractOracle(dspy.Signature):
    """
    Extract the oracle value f*(text) from a text block according to the rubric.

    This is a DSPy-based surrogate oracle for when the true oracle is expensive.
    """
    rubric = dspy.InputField(
        prefix="Oracle definition and output format:\n",
        format=str
    )
    text = dspy.InputField(
        prefix="Text to analyze:\n",
        format=str
    )
    oracle_value = dspy.OutputField(
        prefix="Extracted oracle value:\n",
        format=str
    )

class OracleExtractor(dspy.Module):
    """DSPy module for extracting oracle values."""

    def __init__(self, lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.lm = lm
        self.predictor = dspy.ChainOfThought(ExtractOracle)

    def forward(self, rubric_desc: str, text: str) -> dspy.Prediction:
        with dspy.settings.context(lm=self.lm):
            result = self.predictor(
                rubric=rubric_desc,
                text=text
            )
        return result
```

---

## 5. Oracle System

### 5.1 OracleRouter

```python
from typing import Any, Optional
import logging

class OracleRouter:
    """
    Routes oracle queries to either expensive ground-truth or cheap surrogate.
    Manages caching and tracks usage statistics.
    """

    def __init__(self, rubric: Rubric, use_surrogate_default: bool = False):
        self.rubric = rubric
        self.use_surrogate_default = use_surrogate_default
        self.cache: Dict[str, Any] = {}
        self.stats = {
            'expensive_calls': 0,
            'surrogate_calls': 0,
            'cache_hits': 0
        }

    def __call__(self, text: str, force_expensive: bool = False) -> Any:
        """Extract oracle value, using cache if available."""
        cache_key = hash(text)

        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]

        use_surrogate = self.use_surrogate_default and not force_expensive

        if use_surrogate:
            self.stats['surrogate_calls'] += 1
        else:
            self.stats['expensive_calls'] += 1

        value = self.rubric.extract_oracle(text, use_surrogate=use_surrogate)
        self.cache[cache_key] = value
        return value

    def clear_cache(self):
        self.cache.clear()

    def get_usage_stats(self) -> Dict[str, int]:
        return self.stats.copy()
```

---

## 6. Audit Engine

### 6.1 AuditEngine Implementation

```python
import random
import math
from typing import List, Tuple, Dict

class AuditResult:
    """Result of a probabilistic audit."""

    def __init__(self, passed: bool, violations: List[TreeNode],
                 sample_size: int, total_nodes: int):
        self.passed = passed
        self.violations = violations
        self.sample_size = sample_size
        self.total_nodes = total_nodes
        self.violation_rate = len(violations) / sample_size if sample_size > 0 else 0

    def __repr__(self):
        status = "PASSED" if self.passed else "FAILED"
        return f"AuditResult({status}, {len(self.violations)}/{self.sample_size} violations)"

class AuditEngine:
    """
    Probabilistic audit engine implementing Theorem 1 from the OPS paper.

    Samples m nodes uniformly and checks each for consistency violations.
    """

    def __init__(self, oracle_router: OracleRouter, rubric: Rubric):
        self.oracle = oracle_router
        self.rubric = rubric
        self.logger = logging.getLogger(__name__)

    def compute_sample_size(self, epsilon: float, delta: float) -> int:
        """
        Theorem 1: m = ⌈(2/ε²) ln(2/δ)⌉

        Args:
            epsilon: Maximum tolerable violation fraction
            delta: Confidence parameter (audit fails with prob ≤ δ)

        Returns:
            Required sample size
        """
        m = math.ceil((2 / (epsilon ** 2)) * math.log(2 / delta))
        return m

    def sample_nodes(self, tree: ReductionTree, sample_size: int) -> List[TreeNode]:
        """Uniformly sample nodes from the tree."""
        all_nodes = list(tree.nodes.values())
        return random.sample(all_nodes, min(sample_size, len(all_nodes)))

    def check_c1_sufficiency(self, node: TreeNode) -> bool:
        """Check C1: f*(d_i) = f*(s_i) for leaf node."""
        if not node.is_leaf:
            return True  # C1 only applies to leaves

        oracle_raw = self.oracle(node.raw_text_span, force_expensive=True)
        oracle_summary = self.oracle(node.summary, force_expensive=True)

        return self.rubric.values_match(oracle_raw, oracle_summary)

    def check_c2_idempotence(self, node: TreeNode) -> bool:
        """Check C2: f*(s_j) = f*(σ(s_j))"""
        # Re-summarize the summary
        # For leaf: use LeafSummarizer on summary
        # For internal: use MergeSummarizer on summary
        # Then check if oracle values match

        # This requires access to summarizers, so we'll implement this
        # as a method that takes the summarizers as parameters
        # For now, return True as placeholder
        return True  # Implemented in full version

    def check_c3_merge_consistency(self, node: TreeNode) -> bool:
        """Check C3: f*(σ(s_L, s_R)) = f*(s_L ∪ s_R)"""
        if node.is_leaf:
            return True  # C3 only applies to internal nodes

        # Get oracle of current merged summary
        oracle_merged = self.oracle(node.summary, force_expensive=True)

        # Get oracle of concatenated child summaries
        concat_text = node.left_child.summary + " " + node.right_child.summary
        oracle_concat = self.oracle(concat_text, force_expensive=True)

        return self.rubric.values_match(oracle_merged, oracle_concat)

    def audit_node(self, node: TreeNode) -> Tuple[bool, Optional[ViolationType]]:
        """
        Check a single node for all applicable consistency conditions.

        Returns:
            (passed, violation_type)
        """
        # Check C1 (Sufficiency) for leaves
        if node.is_leaf:
            if not self.check_c1_sufficiency(node):
                return False, ViolationType.C1_SUFFICIENCY

        # Check C3 (Merge Consistency) for internal nodes
        if not node.is_leaf:
            if not self.check_c3_merge_consistency(node):
                return False, ViolationType.C3_MERGE_CONSISTENCY

        # Check C2 (Idempotence) for all nodes
        if not self.check_c2_idempotence(node):
            return False, ViolationType.C2_IDEMPOTENCE

        return True, None

    def run_audit(self, tree: ReductionTree,
                  epsilon: float = 0.1,
                  delta: float = 0.05) -> AuditResult:
        """
        Run probabilistic audit on the reduction tree.

        Args:
            tree: The reduction tree to audit
            epsilon: Maximum tolerable violation fraction (default: 10%)
            delta: Confidence parameter (default: 5% failure probability)

        Returns:
            AuditResult with pass/fail status and violation details
        """
        sample_size = self.compute_sample_size(epsilon, delta)
        sampled_nodes = self.sample_nodes(tree, sample_size)

        violations = []

        for node in sampled_nodes:
            passed, violation_type = self.audit_node(node)

            if not passed:
                node.audit_status = AuditStatus(f"violated_{violation_type.value}")
                node.violation_type = violation_type
                violations.append(node)
                self.logger.warning(
                    f"Node {node.node_id} violated {violation_type.value}"
                )
            else:
                node.audit_status = AuditStatus.PASSED

        audit_passed = len(violations) == 0

        result = AuditResult(
            passed=audit_passed,
            violations=violations,
            sample_size=sample_size,
            total_nodes=len(tree.nodes)
        )

        self.logger.info(
            f"Audit complete: {result} (ε={epsilon}, δ={delta})"
        )

        return result
```

---

## 7. Reduction Pipeline

### 7.1 DocumentChunker

```python
from typing import List
from src.utils.ArticleTextProcessing import ArticleTextProcessing

class DocumentChunker:
    """
    Split document into leaf-sized chunks for tree construction.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Args:
            chunk_size: Target words per chunk
            overlap: Overlapping words between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: str) -> List[str]:
        """Split document into chunks."""
        words = document.split()
        chunks = []

        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(' '.join(chunk_words))
            i += (self.chunk_size - self.overlap)

        return chunks

    def chunk_to_power_of_two(self, document: str) -> List[str]:
        """
        Chunk document such that number of chunks is a power of 2.
        This creates a balanced binary tree.
        """
        chunks = self.chunk_document(document)
        n = len(chunks)

        # Find next power of 2
        target = 2 ** math.ceil(math.log2(n))

        # If we need more chunks, split some larger ones
        while len(chunks) < target:
            # Find the largest chunk and split it
            largest_idx = max(range(len(chunks)), key=lambda i: len(chunks[i]))
            large_chunk = chunks[largest_idx]
            words = large_chunk.split()
            mid = len(words) // 2
            chunks[largest_idx] = ' '.join(words[:mid])
            chunks.insert(largest_idx + 1, ' '.join(words[mid:]))

        return chunks[:target]
```

### 7.2 TreeBuilder

```python
class TreeBuilder:
    """
    Construct reduction tree from document chunks.
    """

    def __init__(self, leaf_summarizer: LeafSummarizer,
                 merge_summarizer: MergeSummarizer,
                 rubric: Rubric):
        self.leaf_summarizer = leaf_summarizer
        self.merge_summarizer = merge_summarizer
        self.rubric = rubric

    def build_tree(self, chunks: List[str]) -> ReductionTree:
        """
        Build complete binary reduction tree from document chunks.

        Algorithm:
        1. Create leaf nodes from chunks
        2. Summarize each leaf (C1: Sufficiency)
        3. Merge pairs bottom-up (C3: Merge Consistency)
        4. Return tree with root summary
        """
        # Create leaf nodes
        leaf_nodes = []
        for idx, chunk in enumerate(chunks):
            node = TreeNode(
                node_id=f"leaf_{idx}",
                is_leaf=True,
                depth=0,
                raw_text_span=chunk
            )

            # Summarize leaf
            result = self.leaf_summarizer(
                rubric_desc=self.rubric.description,
                raw_text=chunk
            )
            node.summary = result.summary
            node.reasoning = result.get('rationale', '')

            leaf_nodes.append(node)

        # Build tree bottom-up
        current_level = leaf_nodes
        depth = 1

        while len(current_level) > 1:
            next_level = []

            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else None

                if right is None:
                    # Odd number of nodes, promote left to next level
                    next_level.append(left)
                    continue

                # Create parent node
                parent = TreeNode(
                    node_id=f"node_{depth}_{i//2}",
                    is_leaf=False,
                    depth=depth,
                    left_child=left,
                    right_child=right
                )

                left.parent = parent
                right.parent = parent

                # Merge summaries
                result = self.merge_summarizer(
                    rubric_desc=self.rubric.description,
                    left=left.summary,
                    right=right.summary
                )
                parent.summary = result.merged_summary
                parent.reasoning = result.get('rationale', '')

                next_level.append(parent)

            current_level = next_level
            depth += 1

        root = current_level[0]
        return ReductionTree(root)
```

### 7.3 ReductionPipeline

```python
class ReductionPipeline:
    """
    End-to-end pipeline: Document → Chunks → Tree → Summary
    """

    def __init__(self,
                 rubric: Rubric,
                 lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 chunk_size: int = 500):
        self.rubric = rubric
        self.chunker = DocumentChunker(chunk_size=chunk_size)

        # Initialize DSPy modules
        self.leaf_summarizer = LeafSummarizer(lm)
        self.merge_summarizer = MergeSummarizer(lm)
        self.oracle_extractor = OracleExtractor(lm)

        # Initialize oracle router (using DSPy extractor as surrogate)
        self.rubric.surrogate_function = lambda text: self.oracle_extractor(
            self.rubric.description, text
        ).oracle_value
        self.oracle_router = OracleRouter(rubric, use_surrogate_default=True)

        # Initialize tree builder and audit engine
        self.tree_builder = TreeBuilder(
            self.leaf_summarizer,
            self.merge_summarizer,
            self.rubric
        )
        self.audit_engine = AuditEngine(self.oracle_router, self.rubric)

    def process_document(self, document: str,
                        run_audit: bool = True,
                        epsilon: float = 0.1,
                        delta: float = 0.05) -> Tuple[ReductionTree, Optional[AuditResult]]:
        """
        Process document through full reduction pipeline.

        Args:
            document: Raw document text
            run_audit: Whether to run probabilistic audit
            epsilon: Audit tolerance parameter
            delta: Audit confidence parameter

        Returns:
            (ReductionTree, AuditResult or None)
        """
        # Step 1: Chunk document
        chunks = self.chunker.chunk_to_power_of_two(document)

        # Step 2: Build reduction tree
        tree = self.tree_builder.build_tree(chunks)

        # Step 3: Run audit (optional)
        audit_result = None
        if run_audit:
            audit_result = self.audit_engine.run_audit(tree, epsilon, delta)

        return tree, audit_result

    def get_summary(self, document: str) -> str:
        """Convenience method to get final summary."""
        tree, _ = self.process_document(document, run_audit=False)
        return tree.get_summary()
```

---

## 8. Bootstrap Optimizer

### 8.1 ConsistencyBootstrap

```python
from typing import List, Tuple
import dspy.teleprompt

class ConsistencyBootstrap:
    """
    Bootstrap optimization using audit violations as training signal.

    Algorithm:
    1. Build tree with current summarizers
    2. Run audit and collect violations
    3. Create training examples from violations:
       - Positive: Non-violated nodes
       - Negative: Violated nodes with corrections
    4. Optimize DSPy modules using violations
    5. Repeat until audit passes
    """

    def __init__(self, pipeline: ReductionPipeline, max_iterations: int = 5):
        self.pipeline = pipeline
        self.max_iterations = max_iterations
        self.training_history = []

    def create_training_example_from_violation(self,
                                               node: TreeNode) -> dspy.Example:
        """
        Convert a violation into a training example.

        For C1 violation: Train LeafSummarizer to preserve f*(raw_text)
        For C3 violation: Train MergeSummarizer to preserve f*(concat)
        """
        if node.violation_type == ViolationType.C1_SUFFICIENCY:
            # Leaf violation: summary doesn't preserve oracle
            true_oracle = self.pipeline.oracle_router(
                node.raw_text_span,
                force_expensive=True
            )

            return dspy.Example(
                rubric=self.pipeline.rubric.description,
                raw_text=node.raw_text_span,
                summary=node.summary,
                expected_oracle=str(true_oracle)
            ).with_inputs("rubric", "raw_text")

        elif node.violation_type == ViolationType.C3_MERGE_CONSISTENCY:
            # Merge violation: merged summary doesn't preserve oracle
            concat_text = (node.left_child.summary + " " +
                          node.right_child.summary)
            true_oracle = self.pipeline.oracle_router(
                concat_text,
                force_expensive=True
            )

            return dspy.Example(
                rubric=self.pipeline.rubric.description,
                left_summary=node.left_child.summary,
                right_summary=node.right_child.summary,
                merged_summary=node.summary,
                expected_oracle=str(true_oracle)
            ).with_inputs("rubric", "left_summary", "right_summary")

        return None

    def optimize_summarizers(self, violations: List[TreeNode]):
        """
        Use violations to optimize DSPy modules via bootstrapping.
        """
        # Separate violations by type
        c1_violations = [v for v in violations
                        if v.violation_type == ViolationType.C1_SUFFICIENCY]
        c3_violations = [v for v in violations
                        if v.violation_type == ViolationType.C3_MERGE_CONSISTENCY]

        # Create training examples
        leaf_examples = [self.create_training_example_from_violation(v)
                        for v in c1_violations]
        merge_examples = [self.create_training_example_from_violation(v)
                         for v in c3_violations]

        # Optimize LeafSummarizer
        if leaf_examples:
            teleprompter = dspy.teleprompt.BootstrapFewShot(
                metric=self._leaf_metric,
                max_bootstrapped_demos=4
            )
            self.pipeline.leaf_summarizer = teleprompter.compile(
                self.pipeline.leaf_summarizer,
                trainset=leaf_examples
            )

        # Optimize MergeSummarizer
        if merge_examples:
            teleprompter = dspy.teleprompt.BootstrapFewShot(
                metric=self._merge_metric,
                max_bootstrapped_demos=4
            )
            self.pipeline.merge_summarizer = teleprompter.compile(
                self.pipeline.merge_summarizer,
                trainset=merge_examples
            )

    def _leaf_metric(self, example, pred, trace=None):
        """Metric for leaf summarizer: does summary preserve oracle?"""
        # Extract oracle from summary
        oracle_summary = self.pipeline.oracle_router(pred.summary)
        oracle_expected = example.expected_oracle
        return self.pipeline.rubric.values_match(oracle_summary, oracle_expected)

    def _merge_metric(self, example, pred, trace=None):
        """Metric for merge summarizer: does merged summary preserve oracle?"""
        oracle_merged = self.pipeline.oracle_router(pred.merged_summary)
        oracle_expected = example.expected_oracle
        return self.pipeline.rubric.values_match(oracle_merged, oracle_expected)

    def bootstrap_loop(self, document: str,
                      epsilon: float = 0.1,
                      delta: float = 0.05) -> ReductionTree:
        """
        Run bootstrap optimization loop until audit passes.

        Args:
            document: Document to process
            epsilon: Audit tolerance
            delta: Audit confidence

        Returns:
            Final reduction tree that passes audit
        """
        for iteration in range(self.max_iterations):
            print(f"\n=== Bootstrap Iteration {iteration + 1}/{self.max_iterations} ===")

            # Build tree and run audit
            tree, audit_result = self.pipeline.process_document(
                document,
                run_audit=True,
                epsilon=epsilon,
                delta=delta
            )

            print(f"Audit result: {audit_result}")
            self.training_history.append({
                'iteration': iteration,
                'audit_result': audit_result,
                'violation_count': len(audit_result.violations)
            })

            # Check if audit passed
            if audit_result.passed:
                print("✓ Audit passed! Bootstrap complete.")
                return tree

            # Optimize using violations
            print(f"Optimizing from {len(audit_result.violations)} violations...")
            self.optimize_summarizers(audit_result.violations)

        print(f"⚠ Bootstrap did not converge after {self.max_iterations} iterations")
        return tree
```

---

## 9. Project Structure

```
ThinkingTrees/
├── src/
│   ├── dataclass/
│   │   ├── __init__.py
│   │   ├── tree_node.py          # TreeNode, AuditStatus, ViolationType
│   │   ├── reduction_tree.py     # ReductionTree
│   │   ├── rubric.py             # Rubric
│   │   └── interface.py          # (reuse from OmniThink)
│   │
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── leaf_summarizer.py    # LeafSummarize, LeafSummarizer
│   │   ├── merge_summarizer.py   # MergeSummarize, MergeSummarizer
│   │   └── oracle_extractor.py   # ExtractOracle, OracleExtractor
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── oracle_router.py      # OracleRouter
│   │   ├── audit_engine.py       # AuditEngine, AuditResult
│   │   ├── tree_builder.py       # TreeBuilder
│   │   ├── document_chunker.py   # DocumentChunker
│   │   └── reduction_pipeline.py # ReductionPipeline
│   │
│   ├── optimization/
│   │   ├── __init__.py
│   │   └── consistency_bootstrap.py  # ConsistencyBootstrap
│   │
│   ├── rubrics/
│   │   ├── __init__.py
│   │   ├── classification.py     # Classification rubric examples
│   │   ├── entity_extraction.py  # Entity extraction rubric
│   │   ├── counting.py           # Counting rubric
│   │   └── custom.py             # User-defined rubrics
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── text_processing.py    # (reuse from OmniThink)
│   │   └── visualization.py      # Tree visualization tools
│   │
│   └── tools/
│       ├── __init__.py
│       ├── lm.py                 # (reuse from OmniThink)
│       └── rm.py                 # (reuse from OmniThink)
│
├── examples/
│   ├── quickstart.py
│   ├── classification_example.py
│   ├── entity_extraction_example.py
│   └── custom_rubric_example.py
│
├── tests/
│   ├── test_tree_node.py
│   ├── test_reduction_tree.py
│   ├── test_summarizers.py
│   ├── test_audit_engine.py
│   ├── test_pipeline.py
│   └── test_bootstrap.py
│
├── notebooks/
│   ├── 01_introduction.ipynb
│   ├── 02_basic_usage.ipynb
│   ├── 03_custom_rubrics.ipynb
│   └── 04_bootstrap_optimization.ipynb
│
├── app.py                        # Streamlit demo app
├── requirements.txt
├── setup.py
├── README.md
└── doc/
    ├── main.pdf                  # OPS paper
    ├── cld_implementation_plan.md  # This document
    └── api_reference.md
```

---

## 10. Implementation Phases

### **Phase 1: Core Data Structures (Week 1)**

**Goal**: Implement the foundational tree structure.

**Tasks**:
- [ ] Implement `TreeNode` with all fields and methods
- [ ] Implement `ReductionTree` with indexing and traversal
- [ ] Implement `Rubric` dataclass
- [ ] Write unit tests for data structures
- [ ] Add NetworkX visualization support

**Deliverables**:
- `src/dataclass/tree_node.py`
- `src/dataclass/reduction_tree.py`
- `src/dataclass/rubric.py`
- `tests/test_tree_node.py`
- `tests/test_reduction_tree.py`

### **Phase 2: DSPy Summarization Modules (Week 2)**

**Goal**: Build the core LLM-based summarizers.

**Tasks**:
- [ ] Implement `LeafSummarizer` with DSPy ChainOfThought
- [ ] Implement `MergeSummarizer` with DSPy ChainOfThought
- [ ] Implement `OracleExtractor` for surrogate oracle
- [ ] Test modules with simple prompts
- [ ] Tune prompts for initial quality

**Deliverables**:
- `src/modules/leaf_summarizer.py`
- `src/modules/merge_summarizer.py`
- `src/modules/oracle_extractor.py`
- `tests/test_summarizers.py`

### **Phase 3: Oracle System (Week 2-3)**

**Goal**: Implement oracle routing and caching.

**Tasks**:
- [ ] Implement `OracleRouter` with cache
- [ ] Create example rubrics:
  - [ ] Classification rubric
  - [ ] Entity extraction rubric
  - [ ] Counting rubric
- [ ] Test oracle extraction on sample texts
- [ ] Benchmark expensive vs surrogate oracle performance

**Deliverables**:
- `src/engine/oracle_router.py`
- `src/rubrics/classification.py`
- `src/rubrics/entity_extraction.py`
- `src/rubrics/counting.py`

### **Phase 4: Tree Construction Pipeline (Week 3-4)**

**Goal**: Build end-to-end document→tree pipeline.

**Tasks**:
- [ ] Implement `DocumentChunker` with power-of-2 balancing
- [ ] Implement `TreeBuilder` with bottom-up construction
- [ ] Implement `ReductionPipeline` orchestration
- [ ] Test on sample documents (Wikipedia articles, news, papers)
- [ ] Profile performance and optimize chunking

**Deliverables**:
- `src/engine/document_chunker.py`
- `src/engine/tree_builder.py`
- `src/engine/reduction_pipeline.py`
- `tests/test_pipeline.py`

### **Phase 5: Audit Engine (Week 4-5)**

**Goal**: Implement probabilistic consistency auditing.

**Tasks**:
- [ ] Implement `AuditEngine` with sampling logic
- [ ] Implement C1 (Sufficiency) check
- [ ] Implement C3 (Merge Consistency) check
- [ ] Implement C2 (Idempotence) check
- [ ] Implement `compute_sample_size()` from Theorem 1
- [ ] Test audit on trees with planted violations
- [ ] Validate statistical properties (ε, δ)

**Deliverables**:
- `src/engine/audit_engine.py`
- `tests/test_audit_engine.py`
- Audit validation notebook

### **Phase 6: Bootstrap Optimizer (Week 5-6)**

**Goal**: Implement consistency bootstrap training loop.

**Tasks**:
- [ ] Implement `ConsistencyBootstrap` class
- [ ] Implement violation→training example conversion
- [ ] Integrate DSPy BootstrapFewShot teleprompter
- [ ] Implement leaf and merge metrics
- [ ] Test bootstrap loop on documents with violations
- [ ] Benchmark convergence rate

**Deliverables**:
- `src/optimization/consistency_bootstrap.py`
- `tests/test_bootstrap.py`
- Bootstrap convergence analysis

### **Phase 7: Integration & Examples (Week 6-7)**

**Goal**: Create user-facing interfaces and examples.

**Tasks**:
- [ ] Create Streamlit app (`app.py`)
- [ ] Write quickstart example
- [ ] Write classification example
- [ ] Write entity extraction example
- [ ] Write custom rubric tutorial
- [ ] Create Jupyter notebooks for documentation
- [ ] Write API reference documentation

**Deliverables**:
- `app.py`
- `examples/` directory
- `notebooks/` directory
- `doc/api_reference.md`

### **Phase 8: Testing & Benchmarking (Week 7-8)**

**Goal**: Validate correctness and performance.

**Tasks**:
- [ ] Create test suite with 80%+ coverage
- [ ] Benchmark on standard datasets:
  - [ ] CNN/DailyMail (classification)
  - [ ] CoNLL-2003 (entity extraction)
  - [ ] SQuAD (question answering as oracle)
- [ ] Compare oracle-preserving rate vs baselines
- [ ] Profile and optimize bottlenecks
- [ ] Write performance tuning guide

**Deliverables**:
- Complete test suite
- Benchmark results
- Performance report

---

## 11. Integration with OmniThink

### 11.1 Shared Components

ThinkingTrees reuses several components from OmniThink:

**Directly Reusable**:
- `src/tools/lm.py` - Language model wrappers (OpenAIModel_dashscope)
- `src/utils/ArticleTextProcessing.py` - Text processing utilities
- `src/dataclass/interface.py` - Abstract base classes (partially)

**Adaptable**:
- `src/tools/mindmap.py` - MindPoint structure can inform TreeNode design
- `app.py` - Streamlit app structure as template

### 11.2 Conceptual Bridges

| OmniThink Component | ThinkingTrees Equivalent | Relationship |
|---------------------|--------------------------|--------------|
| MindPoint | TreeNode | Structural analog (both tree nodes) |
| MindMap.build_map() | TreeBuilder.build_tree() | Construction algorithm |
| ConceptGenerator | LeafSummarizer | DSPy module for text→concept |
| OutlineGenerationModule | - | No direct analog (expansion only) |
| ArticleGenerationModule | - | No direct analog (expansion only) |
| retriever.retrieve() | oracle_router() | Information extraction |

### 11.3 Bidirectional Workflows

**OmniThink → ThinkingTrees**:
1. Generate article with OmniThink
2. Use ThinkingTrees to create hierarchical summary
3. Audit: Does summary preserve key facts from article?

**ThinkingTrees → OmniThink**:
1. Summarize large corpus with ThinkingTrees
2. Use summaries as knowledge base for OmniThink retrieval
3. Generate new articles from compressed knowledge

---

## 12. Testing Strategy

### 12.1 Unit Tests

**Data Structures**:
- TreeNode creation, parent/child relationships
- ReductionTree indexing, traversal, depth queries
- Rubric oracle extraction and value comparison

**Modules**:
- LeafSummarizer produces non-empty summaries
- MergeSummarizer combines two summaries
- OracleExtractor returns correct type

**Engine**:
- OracleRouter caching and statistics
- DocumentChunker produces balanced chunks
- TreeBuilder creates valid binary trees

### 12.2 Integration Tests

**End-to-End Pipeline**:
- Document → Chunks → Tree → Summary
- Pipeline produces correct tree structure
- Root summary is coherent

**Audit Engine**:
- Audit correctly identifies planted C1 violations
- Audit correctly identifies planted C3 violations
- Sample size formula matches Theorem 1

**Bootstrap Optimizer**:
- Violations create valid training examples
- Optimizer improves summarizer quality
- Loop terminates on audit pass

### 12.3 Benchmark Tests

**Datasets**:
- CNN/DailyMail: Classification (article category)
- CoNLL-2003: Entity extraction (person names)
- SQuAD: Question answering (answer extraction)

**Metrics**:
- **Oracle Preservation Rate**: % of nodes with f*(summary) = f*(text)
- **Compression Ratio**: |summary| / |original|
- **Audit Pass Rate**: % of documents passing ε=0.1, δ=0.05 audit
- **Bootstrap Convergence**: Iterations until audit pass

**Baselines**:
- Naive summarization (no oracle awareness)
- Flat summarization (no hierarchy)
- Human-written summaries (gold standard)

---

## 13. Future Extensions

### 13.1 Advanced Features

**Multi-Way Trees**:
- Extend beyond binary trees to k-ary trees
- Optimize k based on document structure

**Adaptive Chunking**:
- Use semantic segmentation instead of fixed word counts
- Chunk at paragraph/section boundaries

**Incremental Updates**:
- Support document edits without full tree rebuild
- Update only affected subtrees

**Parallel Processing**:
- Parallelize leaf summarization across chunks
- Parallelize merge operations at each depth level

### 13.2 Research Directions

**Learned Oracles**:
- Train neural oracle extractors end-to-end
- Multi-task learning across multiple rubrics

**Oracle Discovery**:
- Automatically infer optimal rubrics from document collection
- Cluster documents by oracle similarity

**Guaranteed Summarization**:
- Extend to other guarantees (e.g., faithfulness, diversity)
- Combine with attribution/citation tracking

**Interactive Auditing**:
- Human-in-the-loop audit for ambiguous cases
- Active learning to prioritize audit samples

---

## Appendices

### A. Example Rubrics

#### A.1 Classification Rubric

```python
from src.dataclass.rubric import Rubric

def sentiment_oracle(text: str) -> str:
    """Expensive oracle: Call GPT-4 for sentiment."""
    # Actual implementation would call LLM
    # For demo purposes, simplified
    positive_words = ['good', 'great', 'excellent', 'love']
    negative_words = ['bad', 'terrible', 'hate', 'awful']

    text_lower = text.lower()
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"

sentiment_rubric = Rubric(
    name="sentiment_classification",
    description="Classify the overall sentiment of the text as positive, negative, or neutral.",
    output_type=str,
    oracle_function=sentiment_oracle,
    comparison_function=lambda a, b: a == b,
    task_description="Determine whether the text expresses positive, negative, or neutral sentiment.",
    examples=[
        {"text": "This movie was amazing!", "oracle": "positive"},
        {"text": "Worst experience ever.", "oracle": "negative"},
        {"text": "The weather is cloudy.", "oracle": "neutral"}
    ]
)
```

#### A.2 Entity Extraction Rubric

```python
def extract_entities_oracle(text: str) -> List[str]:
    """Expensive oracle: Call NER model."""
    # Actual implementation would use spaCy or similar
    import re
    # Simplified: Extract capitalized words
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    return sorted(set(entities))

def entity_list_match(list1: List[str], list2: List[str]) -> bool:
    """Check if two entity lists are equivalent."""
    return set(list1) == set(list2)

entity_rubric = Rubric(
    name="person_name_extraction",
    description="Extract all person names mentioned in the text.",
    output_type=list,
    oracle_function=extract_entities_oracle,
    comparison_function=entity_list_match,
    task_description="Identify and list all person names appearing in the text.",
    examples=[
        {"text": "Alice met Bob at the conference.", "oracle": ["Alice", "Bob"]},
        {"text": "Dr. Smith and Prof. Jones published a paper.", "oracle": ["Smith", "Jones"]}
    ]
)
```

#### A.3 Counting Rubric

```python
def count_events_oracle(text: str) -> int:
    """Count number of events mentioned."""
    # Simplified: Count sentences with action verbs
    import re
    sentences = re.split(r'[.!?]', text)
    action_verbs = ['happened', 'occurred', 'took place', 'began', 'ended']
    count = sum(1 for s in sentences if any(v in s.lower() for v in action_verbs))
    return count

counting_rubric = Rubric(
    name="event_counting",
    description="Count the number of distinct events mentioned in the text.",
    output_type=int,
    oracle_function=count_events_oracle,
    comparison_function=lambda a, b: a == b,
    task_description="Count how many separate events are described in the text.",
    examples=[
        {"text": "The meeting happened on Monday. The launch occurred on Friday.", "oracle": 2}
    ]
)
```

### B. Quick Start Commands

```bash
# Installation
git clone https://github.com/yourusername/ThinkingTrees.git
cd ThinkingTrees
pip install -r requirements.txt

# Set up environment
export OPENAI_API_KEY="your-api-key"

# Run quick start example
python examples/quickstart.py

# Run Streamlit demo
streamlit run app.py

# Run tests
pytest tests/

# Run benchmark
python benchmarks/run_cnn_dailymail.py
```

### C. Quick Start Example

```python
from src.engine.reduction_pipeline import ReductionPipeline
from src.dataclass.rubric import Rubric
from src.tools.lm import OpenAIModel_dashscope
import os

# Define rubric
def simple_oracle(text: str) -> str:
    return "positive" if "good" in text.lower() else "negative"

rubric = Rubric(
    name="simple_sentiment",
    description="Determine if text is positive or negative",
    output_type=str,
    oracle_function=simple_oracle,
    comparison_function=lambda a, b: a == b
)

# Initialize LLM
lm = OpenAIModel_dashscope(
    model='gpt-4o',
    max_tokens=1000,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Create pipeline
pipeline = ReductionPipeline(rubric=rubric, lm=lm)

# Process document
document = """
This is a great product. I really enjoyed using it.
The quality is excellent and the price is reasonable.
I would definitely recommend it to others.
"""

tree, audit_result = pipeline.process_document(document)

print(f"Original length: {len(document)} characters")
print(f"Summary length: {len(tree.get_summary())} characters")
print(f"Summary: {tree.get_summary()}")
print(f"Audit: {audit_result}")
```

---

## Conclusion

This implementation plan provides a comprehensive roadmap for building **ThinkingTrees**, a hierarchical summarization framework with oracle-preserving guarantees. The system inverts and complements the OmniThink architecture, creating a powerful reduction engine that maintains provable consistency properties.

**Key Deliverables**:
1. Complete reduction tree implementation
2. DSPy-based summarization modules
3. Probabilistic audit engine
4. Bootstrap optimization loop
5. Example rubrics and applications
6. Streamlit demo application
7. Comprehensive test suite

**Timeline**: 8 weeks (2 months)

**Next Steps**:
1. Begin Phase 1: Implement core data structures
2. Set up project repository and development environment
3. Create initial test framework
4. Start documenting API as components are built

This plan balances theoretical rigor with practical engineering, ensuring that the final system is both mathematically sound and user-friendly.
