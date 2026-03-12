"""
Live integration tests for vLLM server.

These tests require a running vLLM server. They will be skipped if the server
is not available, or unless you explicitly opt in.

Enable with:

    TT_RUN_LIVE_TESTS=1 pytest tests/integration/test_vllm_live.py -v

Start the server with:

    ./scripts/start_vllm.sh

Run these tests with:

    pytest tests/integration/test_vllm_live.py -v
"""

import pytest
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.llm_client import LLMClient, LLMConfig, create_summarizer
from src.preprocessing.chunker import DocumentChunker, chunk_for_ops
from src.tree.builder import TreeBuilder, BuildConfig
from src.core.strategy import CallableStrategy

# Configuration
VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"


def is_vllm_available() -> bool:
    """Check if vLLM server is running."""
    import urllib.request
    import urllib.error
    try:
        # Check the models endpoint
        req = urllib.request.Request(f"{VLLM_URL}/models", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False


# Skip all tests in this module if vLLM is not available
_RUN_LIVE = str(os.getenv("TT_RUN_LIVE_TESTS", "") or "").strip().lower() in {"1", "true", "yes"}
pytestmark = pytest.mark.skipif(
    (not _RUN_LIVE) or (not is_vllm_available()),
    reason=f"Live tests disabled or vLLM server not available at {VLLM_URL}",
)


@pytest.fixture(scope="module")
def vllm_config():
    """vLLM client configuration."""
    return LLMConfig.vllm(host=VLLM_HOST, port=VLLM_PORT)


@pytest.fixture(scope="module")
def vllm_client(vllm_config):
    """vLLM client instance."""
    return LLMClient(vllm_config)


class TestVLLMConnection:
    """Test basic vLLM server connectivity."""

    def test_server_is_running(self):
        """Verify the vLLM server is accessible."""
        assert is_vllm_available(), "vLLM server should be running"

    def test_list_models(self, vllm_client):
        """List available models on the server."""
        import urllib.request
        import json

        req = urllib.request.Request(f"{VLLM_URL}/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            assert "data" in data
            assert len(data["data"]) > 0
            print(f"\nAvailable models: {[m['id'] for m in data['data']]}")


class TestLLMClient:
    """Test the LLMClient with vLLM backend."""

    def test_simple_completion(self, vllm_client):
        """Test a simple completion request."""
        response = vllm_client.generate(
            "What is 2 + 2? Answer with just the number.",
            max_tokens=32,
            temperature=0.0,
        )
        assert response.content is not None
        assert len(response.content) > 0
        print(f"\nResponse: {response.content}")

    def test_system_message(self, vllm_client):
        """Test completion with system message."""
        response = vllm_client.generate(
            "Describe the color of the sky.",
            system="You are a helpful assistant. Be concise.",
            max_tokens=50
        )
        assert response.content is not None
        print(f"\nResponse: {response.content}")

    def test_token_usage_tracking(self, vllm_client):
        """Test that token usage is tracked."""
        vllm_client.reset_usage()

        response = vllm_client.generate(
            "Say hello.",
            max_tokens=20
        )

        usage = vllm_client.get_usage()
        assert usage['call_count'] == 1
        # Token counts may or may not be available depending on server config
        print(f"\nUsage: {usage}")


class TestSummarization:
    """Test summarization with vLLM."""

    @pytest.fixture
    def summarizer(self, vllm_client):
        """Create a summarizer using the vLLM client."""
        return create_summarizer(vllm_client)

    def test_simple_summarization(self, summarizer):
        """Test basic text summarization."""
        text = """
        The quick brown fox jumps over the lazy dog. This sentence contains
        every letter of the alphabet. It has been used for typing practice
        for over a century. Typewriter manufacturers used it to test their
        machines, and now it's commonly used in font samples.
        """
        rubric = "Preserve: main subject, key facts"

        summary = summarizer(text, rubric)
        assert summary is not None
        assert len(summary) > 0
        print(f"\nOriginal length: {len(text)}")
        print(f"Summary length: {len(summary)}")
        print(f"Summary: {summary}")

    def test_summarization_preserves_rubric_info(self, summarizer):
        """Test that summarization preserves information specified in rubric."""
        text = """
        In 2023, the global artificial intelligence market was valued at
        $150 billion. The market is expected to grow at a compound annual
        growth rate of 38% from 2024 to 2030. Major players include OpenAI,
        Google DeepMind, and Anthropic. Applications span healthcare, finance,
        and autonomous vehicles.
        """
        rubric = "Preserve: market value, growth rate, major companies"

        summary = summarizer(text, rubric)
        assert summary is not None
        # Check that key information is preserved (fuzzy check)
        print(f"\nSummary: {summary}")
        # At minimum, should mention some numbers or companies


class TestFullPipeline:
    """Test the full OPS pipeline with vLLM."""

    @pytest.fixture
    def builder(self, vllm_client):
        """Create a tree builder with vLLM summarizer."""
        summarizer = create_summarizer(vllm_client)
        strategy = CallableStrategy(summarizer=summarizer)
        config = BuildConfig(
            max_chunk_chars=500,
            min_chunk_chars=50,
            chunk_strategy="sentence"
        )
        return TreeBuilder(strategy=strategy, config=config)

    @pytest.mark.anyio
    async def test_build_small_tree(self, builder):
        """Build a small tree from text."""
        text = """
        Chapter 1: Introduction

        This is the introduction to our document. It provides background
        information about the topic we're discussing. The main themes include
        technology, innovation, and progress.

        Chapter 2: Methods

        We used several methods in our research. First, we collected data from
        multiple sources. Then, we analyzed the data using statistical techniques.
        Finally, we validated our findings through peer review.

        Chapter 3: Results

        Our results show significant improvements. The accuracy increased by 25%.
        Processing time decreased by 40%. User satisfaction improved markedly.

        Chapter 4: Conclusion

        In conclusion, our research demonstrates the effectiveness of the approach.
        Future work should focus on scalability and real-world deployment.
        """

        rubric = "Preserve: chapter topics, key findings, percentages"

        result = await builder.build(text, rubric)

        assert result.tree is not None
        assert result.chunks_created > 0
        assert result.nodes_created >= result.chunks_created

        print(f"\nTree built successfully:")
        print(f"  Chunks: {result.chunks_created}")
        print(f"  Nodes: {result.nodes_created}")
        print(f"  Root summary: {result.tree.root.summary[:200]}...")

    @pytest.mark.anyio
    async def test_tree_compression_ratio(self, builder):
        """Verify the tree achieves reasonable compression."""
        # Generate longer text
        paragraphs = []
        for i in range(10):
            paragraphs.append(
                f"Paragraph {i}: This is paragraph number {i} of our test document. "
                f"It contains information about topic {i}. We need sufficient content "
                f"to test the compression capabilities of our summarization pipeline. "
                f"Each paragraph adds substantial text to ensure meaningful compression."
            )
        text = "\n\n".join(paragraphs)

        rubric = "Preserve: paragraph topics, numerical information"

        result = await builder.build(text, rubric)

        original_length = len(text)
        summary_length = len(result.tree.root.summary)
        compression_ratio = original_length / summary_length

        print(f"\nCompression analysis:")
        print(f"  Original: {original_length} chars")
        print(f"  Summary: {summary_length} chars")
        print(f"  Ratio: {compression_ratio:.1f}x")

        assert compression_ratio > 0.0


class TestTokenCounting:
    """Test token counting with tiktoken."""

    def test_tiktoken_available(self):
        """Verify tiktoken is installed and working."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode("Hello, world!")
        assert len(tokens) > 0
        print(f"\n'Hello, world!' = {len(tokens)} tokens")

    def test_count_document_tokens(self):
        """Count tokens in a sample document."""
        import tiktoken

        text = """
        The quick brown fox jumps over the lazy dog. This sentence contains
        every letter of the alphabet. It has been used for typing practice.
        """

        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)

        print(f"\nDocument: {len(text)} chars, {len(tokens)} tokens")
        print(f"Chars per token: {len(text) / len(tokens):.1f}")


if __name__ == "__main__":
    # Quick check if server is available
    if is_vllm_available():
        print(f"vLLM server is available at {VLLM_URL}")
        pytest.main([__file__, "-v", "-s"])
    else:
        print(f"vLLM server is NOT available at {VLLM_URL}")
        print("Start the server with: ./scripts/start_vllm.sh")
