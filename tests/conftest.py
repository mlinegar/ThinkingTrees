"""
Shared pytest fixtures for ThinkingTrees tests.

Uses test configuration from tests/config/test_settings.yaml for consistency.
"""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.data_models import Node, Tree, leaf, node
from src.preprocessing.chunker import TextChunk, Chunker
from src.tree.builder import IdentitySummarizer

from tests.test_config import ConfigAccessor, get_sample_text, get_rubric


# --- Test Configuration Fixture ---

@pytest.fixture(scope="session")
def test_cfg():
    """Test configuration accessor."""
    return ConfigAccessor()


# --- Sample Text Fixtures ---

@pytest.fixture
def short_text():
    """Short text that fits in a single chunk."""
    text = get_sample_text("short")
    if text:
        return text
    return "This is a short piece of text. It has just a few sentences."


@pytest.fixture
def medium_text():
    """Medium text requiring a few chunks."""
    text = get_sample_text("medium")
    if text:
        return text
    return """
    The quick brown fox jumps over the lazy dog. This sentence contains every
    letter of the alphabet. It has been used for typing practice for over a
    century now.

    Another paragraph begins here. We need enough text to create multiple
    chunks when testing the chunker. This paragraph adds more content to
    ensure we have sufficient material.

    A third paragraph provides even more content. Testing chunking behavior
    requires text of various lengths. This should be enough for basic tests.
    """


@pytest.fixture
def long_text():
    """Longer text requiring many chunks."""
    paragraphs = []
    for i in range(20):
        paragraphs.append(
            f"Paragraph {i}: This is paragraph number {i} of our test document. "
            f"It contains several sentences about topic {i}. We need enough content "
            f"to thoroughly test the chunking and tree building algorithms. "
            f"Each paragraph adds approximately 200 characters to ensure we have "
            f"sufficient material for testing purposes."
        )
    return "\n\n".join(paragraphs)


@pytest.fixture
def sample_rubric():
    """Basic rubric for testing summarization."""
    rubric = get_rubric("default")
    if rubric:
        return rubric
    return "Preserve: main topics, key entities, numerical data, and conclusions."


@pytest.fixture
def minimal_rubric():
    """Minimal rubric for simple tests."""
    rubric = get_rubric("simple")
    if rubric:
        return rubric
    return "Keep important information."


# --- Mock Summarizer Fixtures ---

@pytest.fixture
def identity_summarizer():
    """Summarizer that returns input unchanged."""
    return IdentitySummarizer()


@pytest.fixture
def counting_summarizer():
    """Summarizer that counts calls and returns abbreviated content."""
    class CountingSummarizer:
        def __init__(self):
            self.calls = []

        def __call__(self, content: str, rubric: str) -> str:
            self.calls.append({'content': content, 'rubric': rubric})
            # Return abbreviated version
            return f"[Summary #{len(self.calls)}] {content[:50]}..."

    return CountingSummarizer()


@pytest.fixture
def mock_llm():
    """Mock LLM that returns predictable responses."""
    class MockLLM:
        def __init__(self):
            self.calls = []
            self.responses = []

        def set_responses(self, responses):
            """Set predefined responses."""
            self.responses = list(responses)

        def complete(self, prompt: str) -> str:
            self.calls.append(prompt)
            if self.responses:
                return self.responses.pop(0)
            return f"Mock response for: {prompt[:30]}..."

        def __call__(self, content: str, rubric: str) -> str:
            return self.complete(f"Content: {content}\nRubric: {rubric}")

    return MockLLM()


# --- Chunker Fixtures ---

@pytest.fixture
def default_chunker():
    """Default chunker with standard settings."""
    return Chunker(max_tokens=500)


@pytest.fixture
def small_chunker():
    """Chunker with small max size for testing."""
    return Chunker(max_tokens=100)


@pytest.fixture
def sample_chunks():
    """Pre-made chunks for testing."""
    return [
        TextChunk(text="First chunk content.", start_char=0, end_char=20, chunk_index=0),
        TextChunk(text="Second chunk content.", start_char=21, end_char=42, chunk_index=1),
        TextChunk(text="Third chunk content.", start_char=43, end_char=63, chunk_index=2),
        TextChunk(text="Fourth chunk content.", start_char=64, end_char=85, chunk_index=3),
    ]


# --- Node Fixtures ---

@pytest.fixture
def sample_leaf():
    """Single leaf node."""
    return leaf("This is leaf content.", node_id="test_leaf")


@pytest.fixture
def sample_leaves():
    """List of 4 leaf nodes."""
    return [
        leaf(f"Leaf {i} content.", node_id=f"leaf_{i}")
        for i in range(4)
    ]


@pytest.fixture
def simple_tree(sample_leaves):
    """Simple 3-level tree with 4 leaves."""
    # Build manually for predictable structure
    # Level 0: 4 leaves
    # Level 1: 2 internal nodes
    # Level 2: 1 root

    left_pair = node(
        sample_leaves[0],
        sample_leaves[1],
        summary="Summary of leaves 0-1",
        node_id="internal_0"
    )

    right_pair = node(
        sample_leaves[2],
        sample_leaves[3],
        summary="Summary of leaves 2-3",
        node_id="internal_1"
    )

    root = node(
        left_pair,
        right_pair,
        summary="Root summary of all leaves",
        node_id="root"
    )

    return Tree(root=root, rubric="Test rubric")


@pytest.fixture
def single_node_tree():
    """Tree with just one node (leaf = root)."""
    single_leaf = leaf("Single node content.", node_id="only_node")
    return Tree(root=single_leaf, rubric="Test rubric")


# --- Utility Fixtures ---

@pytest.fixture
def temp_text_file(tmp_path, medium_text):
    """Temporary text file for file-based tests."""
    file_path = tmp_path / "test_document.txt"
    file_path.write_text(medium_text)
    return file_path


@pytest.fixture
def temp_empty_file(tmp_path):
    """Temporary empty file."""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("")
    return file_path


# --- Collection Guard ---

def pytest_ignore_collect(collection_path: Path, config) -> bool:
    """Skip tests that import the externalized ``treepo._research`` archive.

    The standalone treepo package holds the publishable surface; tests that
    import the research archive stay on disk for reference but are not part
    of this workspace's default test surface.
    """

    del config
    path = Path(collection_path)
    if path.suffix != ".py" or not path.name.startswith("test_"):
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "from treepo._research" in text or "import treepo._research" in text
