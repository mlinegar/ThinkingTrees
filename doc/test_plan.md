# ThinkingTrees Test Plan

## Testing Philosophy

We follow **Test-Driven Development (TDD)**:
1. Write tests that define expected behavior
2. Implement minimal code to pass tests
3. Refactor while maintaining coverage

## Test Categories

### 1. Unit Tests
Test individual components in isolation.

### 2. Integration Tests
Test component interactions (e.g., chunker + builder).

### 3. Property-Based Tests
Test invariants that should always hold (e.g., tree structure properties).

### 4. Mock Tests
Test LLM-dependent code with mocked responses.

---

## Test Specifications by Module

### `tests/core/test_data_models.py`

#### OPSNode Tests

```python
class TestOPSNode:
    def test_create_leaf_node():
        """Leaf nodes have raw_text_span and level=0."""

    def test_create_internal_node():
        """Internal nodes have children and level > 0."""

    def test_is_leaf_property():
        """is_leaf returns True iff no children."""

    def test_node_id_unique():
        """Each node gets a unique ID."""

    def test_leaf_invariants():
        """Leaves: raw_text_span set, no children, level=0."""

    def test_internal_invariants():
        """Internal: children set, level > 0."""
```

#### OPSTree Tests

```python
class TestOPSTree:
    def test_create_single_node_tree():
        """Tree with one leaf (root = leaf)."""

    def test_create_binary_tree():
        """Standard binary tree structure."""

    def test_height_calculation():
        """Tree height is max depth."""

    def test_node_count():
        """Correct total node count."""

    def test_leaf_count():
        """Correct leaf count."""

    def test_traverse_preorder():
        """Preorder traversal visits nodes correctly."""

    def test_traverse_postorder():
        """Postorder traversal visits nodes correctly."""

    def test_traverse_level_order():
        """Level order traversal (BFS)."""

    def test_find_node_by_id():
        """Locate node by ID."""

    def test_get_path_to_root():
        """Path from leaf to root."""
```

### `tests/preprocessing/test_chunker.py`

```python
class TestDocumentChunker:
    def test_chunk_short_text():
        """Text under limit → single chunk."""

    def test_chunk_at_sentence_boundary():
        """Chunks break at sentence ends."""

    def test_chunk_respects_max_chars():
        """No chunk exceeds max_chars."""

    def test_chunk_preserves_all_text():
        """Joining chunks reproduces original."""

    def test_chunk_with_overlap():
        """Overlap between adjacent chunks."""

    def test_chunk_handles_long_word():
        """Word longer than max_chars handled."""

    def test_chunk_empty_text():
        """Empty text → empty list."""

    def test_chunk_whitespace_only():
        """Whitespace-only text → empty list."""

    def test_chunk_preserves_paragraphs():
        """Prefer breaking at paragraphs."""

    def test_chunk_unicode():
        """Handle unicode correctly."""
```

#### TextChunk Tests

```python
class TestTextChunk:
    def test_chunk_text_property():
        """Access chunk text."""

    def test_chunk_char_interval():
        """Character positions in original doc."""

    def test_chunk_token_interval():
        """Token positions if tokenized."""
```

### `tests/ops_engine/test_builder.py`

```python
class TestTreeBuilder:
    def test_build_single_chunk():
        """One chunk → tree with single node (leaf=root)."""

    def test_build_two_chunks():
        """Two chunks → height-1 tree."""

    def test_build_four_chunks():
        """Four chunks → height-2 balanced tree."""

    def test_build_odd_chunks():
        """Odd number of chunks handled correctly."""

    def test_leaf_summaries_set():
        """Leaf nodes have summaries from raw text."""

    def test_internal_summaries_set():
        """Internal nodes have summaries from children."""

    def test_tree_structure_valid():
        """All tree invariants hold."""

    def test_rubric_passed_to_summarizer():
        """Rubric used in summarization calls."""
```

#### Mock LLM Tests

```python
class TestTreeBuilderWithMockLLM:
    def test_summarizer_called_for_each_node():
        """Verify LLM called correct number of times."""

    def test_summarizer_receives_correct_input():
        """Verify input format to LLM."""

    def test_handles_llm_failure():
        """Graceful handling of LLM errors."""
```

### `tests/ops_engine/test_auditor.py`

```python
class TestAuditor:
    def test_audit_samples_within_budget():
        """Number of samples <= budget."""

    def test_audit_checks_leaves():
        """Sufficiency check on leaf nodes."""

    def test_audit_checks_internal():
        """Merge consistency on internal nodes."""

    def test_audit_flags_failures():
        """Failed audits marked on nodes."""

    def test_audit_records_scores():
        """Discrepancy scores recorded."""

    def test_get_failed_nodes():
        """Retrieve all failed nodes."""
```

### `tests/conftest.py` - Shared Fixtures

```python
import pytest

@pytest.fixture
def sample_text():
    """Short sample text for testing."""
    return "This is sentence one. This is sentence two. This is sentence three."

@pytest.fixture
def long_text():
    """Longer text that requires multiple chunks."""
    return " ".join([f"Sentence {i}." for i in range(100)])

@pytest.fixture
def simple_rubric():
    """Basic rubric for testing."""
    return "Preserve: main topics, key entities, numerical data."

@pytest.fixture
def mock_llm():
    """Mock LLM that returns predictable responses."""
    class MockLLM:
        def __init__(self):
            self.calls = []

        def complete(self, prompt):
            self.calls.append(prompt)
            return f"Summary of: {prompt[:50]}..."
    return MockLLM()

@pytest.fixture
def sample_tree():
    """Pre-built sample tree for testing."""
    # Returns a 3-level tree with 4 leaves
    ...

@pytest.fixture
def chunker():
    """Default chunker instance."""
    from src.preprocessing.chunker import DocumentChunker
    return DocumentChunker(max_chunk_chars=100)
```

---

## Test Execution

### Running All Tests
```bash
pytest tests/ -v
```

### Running Specific Module
```bash
pytest tests/core/test_data_models.py -v
```

### Running with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Running Property Tests Only
```bash
pytest tests/ -v -m property
```

---

## Coverage Goals

| Module | Target Coverage |
|--------|-----------------|
| `core/data_models.py` | 95% |
| `preprocessing/chunker.py` | 90% |
| `ops_engine/builder.py` | 85% |
| `ops_engine/auditor.py` | 85% |
| `core/llm_client.py` | 80% (mock-heavy) |

---

## Test Implementation Priority

### Phase 1: Foundation (Immediate)
1. `test_data_models.py` - OPSNode and OPSTree basics
2. `test_chunker.py` - Text chunking
3. `conftest.py` - Shared fixtures

### Phase 2: Core Engine
4. `test_builder.py` - Tree construction
5. `test_builder.py` (mock tests) - LLM integration

### Phase 3: Audit & Optimize
6. `test_auditor.py` - Audit logic
7. `test_optimizer.py` - Bootstrap optimization

---

## Property-Based Testing Ideas

Using `hypothesis`:

```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=10000))
def test_chunk_preserves_content(text):
    """Property: chunking never loses content."""
    chunks = chunker.chunk_text(text)
    reconstructed = "".join(c.text for c in chunks)
    # Allow whitespace normalization
    assert text.split() == reconstructed.split()

@given(st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=20))
def test_tree_has_correct_leaf_count(texts):
    """Property: tree has same number of leaves as input chunks."""
    tree = builder.build(texts)
    assert tree.leaf_count == len(texts)
```
