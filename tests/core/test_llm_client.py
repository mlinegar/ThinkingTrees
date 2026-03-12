"""
Tests for LLM client module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import threading

from src.core.llm_client import (
    ServerType,
    LLMConfig,
    LLMResponse,
    LLMClient,
    MockLLMClient,
    create_client,
    create_summarizer,
    vllm_client,
    sglang_client,
    openai_client,
)


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_config(self):
        """Default config points to local vLLM."""
        config = LLMConfig()
        assert config.base_url == "http://localhost:8000/v1"
        assert config.api_key == "EMPTY"
        assert config.server_type == ServerType.VLLM

    def test_vllm_config(self):
        """vLLM factory method creates correct config."""
        config = LLMConfig.vllm(model="llama-2-7b", port=8080)
        assert "8080" in config.base_url
        assert config.model == "llama-2-7b"
        assert config.server_type == ServerType.VLLM

    def test_sglang_config(self):
        """SGLang factory method creates correct config."""
        config = LLMConfig.sglang(model="mistral", port=30000)
        assert "30000" in config.base_url
        assert config.model == "mistral"
        assert config.server_type == ServerType.SGLANG

    def test_openai_config(self):
        """OpenAI factory method creates correct config."""
        config = LLMConfig.openai(model="gpt-4o", api_key="test-key")
        assert "api.openai.com" in config.base_url
        assert config.model == "gpt-4o"
        assert config.api_key == "test-key"
        assert config.server_type == ServerType.OPENAI

    def test_from_env(self):
        """Config can load from environment."""
        with patch.dict('os.environ', {
            'LLM_BASE_URL': 'http://myserver:9000/v1',
            'LLM_MODEL': 'custom-model',
            'LLM_API_KEY': 'my-key'
        }):
            config = LLMConfig.from_env()
            assert config.base_url == 'http://myserver:9000/v1'
            assert config.model == 'custom-model'
            assert config.api_key == 'my-key'


class TestLLMResponse:
    """Tests for LLMResponse."""

    def test_response_creation(self):
        """Response stores all fields."""
        response = LLMResponse(
            content="Hello world",
            model="gpt-4",
            prompt_tokens=10,
            completion_tokens=5
        )
        assert response.content == "Hello world"
        assert response.model == "gpt-4"
        assert response.total_tokens == 15

    def test_default_values(self):
        """Response has sensible defaults."""
        response = LLMResponse(content="test", model="test")
        assert response.prompt_tokens == 0
        assert response.completion_tokens == 0
        assert response.raw_response is None


class TestMockLLMClient:
    """Tests for MockLLMClient."""

    def test_default_response(self):
        """Mock returns default response."""
        client = MockLLMClient()
        response = client("Hello")
        assert "Response to:" in response or "Summary:" in response

    def test_custom_response_function(self):
        """Custom response function works."""
        client = MockLLMClient(response_fn=lambda x: f"CUSTOM: {x}")
        response = client("test input")
        assert response == "CUSTOM: test input"

    def test_tracks_calls(self):
        """Mock tracks all calls."""
        client = MockLLMClient()
        client("first")
        client("second")
        client("third")
        assert len(client.calls) == 3
        assert client.calls[0] == "first"

    def test_reset_clears_calls(self):
        """Reset clears call history."""
        client = MockLLMClient()
        client("test")
        client.reset()
        assert len(client.calls) == 0

    def test_generate_returns_response(self):
        """generate() returns full LLMResponse."""
        client = MockLLMClient()
        response = client.generate("test prompt")
        assert isinstance(response, LLMResponse)
        assert response.model == "mock"
        assert response.content is not None

    def test_chat_interface(self):
        """Mock handles chat-style messages."""
        client = MockLLMClient()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
        response = client.chat(messages)
        assert isinstance(response, LLMResponse)

    def test_get_usage(self):
        """Mock tracks usage."""
        client = MockLLMClient()
        client("test")
        client("test2")
        usage = client.get_usage()
        assert usage['call_count'] == 2


class TestLLMClient:
    """Tests for LLMClient."""

    def test_initialization(self):
        """Client initializes with config."""
        config = LLMConfig.vllm(model="test")
        client = LLMClient(config)
        assert client.config.model == "test"

    @patch('openai.OpenAI')
    def test_generate(self, mock_openai_class):
        """Generate works with mocked OpenAI."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.model = "test-model"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient(LLMConfig())
        response = client.generate("Hello")

        assert response.content == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch('openai.OpenAI')
    def test_chat(self, mock_openai_class):
        """Chat works with message list."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.model = "test"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient(LLMConfig())
        messages = [{"role": "user", "content": "Hi"}]
        response = client.chat(messages)

        assert response.content == "Response"

    @patch('openai.OpenAI')
    def test_usage_tracking(self, mock_openai_class):
        """Client tracks usage across calls."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.model = "test"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient(LLMConfig())
        client.generate("test1")
        client.generate("test2")

        usage = client.get_usage()
        assert usage['call_count'] == 2
        assert usage['prompt_tokens'] == 20
        assert usage['completion_tokens'] == 10

    @patch('openai.OpenAI')
    def test_reset_usage(self, mock_openai_class):
        """Usage can be reset."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.model = "test"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient(LLMConfig())
        client.generate("test")

        old_usage = client.reset_usage()
        assert old_usage['call_count'] == 1

        new_usage = client.get_usage()
        assert new_usage['call_count'] == 0


class TestCreateClient:
    """Tests for create_client factory."""

    def test_create_real_client(self):
        """Factory creates LLMClient by default."""
        client = create_client()
        assert isinstance(client, LLMClient)

    def test_create_mock_client(self):
        """Factory creates MockLLMClient when mock=True."""
        client = create_client(mock=True)
        assert isinstance(client, MockLLMClient)

    def test_with_config(self):
        """Factory accepts config."""
        config = LLMConfig.sglang(model="test")
        client = create_client(config)
        assert client.config.model == "test"


class TestConvenienceClients:
    """Tests for convenience client functions."""

    def test_vllm_client(self):
        """vllm_client creates correct config."""
        client = vllm_client(model="llama", port=9000)
        assert isinstance(client, LLMClient)
        assert client.config.server_type == ServerType.VLLM
        assert "9000" in client.config.base_url

    def test_sglang_client(self):
        """sglang_client creates correct config."""
        client = sglang_client(model="mistral")
        assert isinstance(client, LLMClient)
        assert client.config.server_type == ServerType.SGLANG
        assert "30000" in client.config.base_url

    def test_openai_client(self):
        """openai_client creates correct config."""
        client = openai_client(model="gpt-4o", api_key="test")
        assert isinstance(client, LLMClient)
        assert client.config.server_type == ServerType.OPENAI
        assert client.config.api_key == "test"


class TestCreateSummarizer:
    """Tests for create_summarizer."""

    def test_creates_callable(self):
        """Factory creates callable summarizer."""
        summarizer = create_summarizer()
        assert callable(summarizer)

    def test_summarizer_uses_client(self):
        """Summarizer uses provided client."""
        mock_client = MockLLMClient(response_fn=lambda x: "SUMMARY")
        summarizer = create_summarizer(client=mock_client)

        result = summarizer("Some content", "Keep facts")
        assert result == "SUMMARY"
        assert len(mock_client.calls) == 1

    def test_summarizer_includes_rubric(self):
        """Summarizer passes rubric in prompt."""
        mock_client = MockLLMClient(response_fn=lambda x: x)
        summarizer = create_summarizer(client=mock_client)

        result = summarizer("Content here", "Important rubric!")
        assert "Important rubric!" in result
        assert "Content here" in result


class TestRetryLogic:
    """Tests for retry behavior."""

    @patch('openai.OpenAI')
    def test_retry_on_failure(self, mock_openai_class):
        """Client retries on failure."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        call_count = 0

        def failing_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Simulated failure")
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "Success"
            response.model = "test"
            response.usage = None
            return response

        mock_client.chat.completions.create.side_effect = failing_then_success

        config = LLMConfig(retry_delay=0.01)
        client = LLMClient(config)
        result = client.generate("test")

        assert result.content == "Success"
        assert call_count == 3


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_mock_calls(self):
        """Mock client is thread-safe."""
        client = MockLLMClient()
        num_threads = 10
        calls_per_thread = 100

        def make_calls():
            for _ in range(calls_per_thread):
                client("test")

        threads = [threading.Thread(target=make_calls) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        usage = client.get_usage()
        assert usage['call_count'] == num_threads * calls_per_thread


class TestIntegrationWithBuilder:
    """Tests for integration with tree builder."""

    def test_summarizer_compatible_with_build(self):
        """Summarizer works with build() convenience function."""
        from src.tree.builder import build

        summarizer = create_summarizer()

        tree = build(
            "This is some test content that needs to be summarized. " * 5,
            summarizer=summarizer,
            rubric="Preserve key facts",
            max_chars=500
        )

        assert tree is not None
        assert tree.validate() == []
