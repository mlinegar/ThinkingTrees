"""
LLM client for OPS (Oracle-Preserving Summarization).

Provides a unified OpenAI-compatible client that works with:
- vLLM (default local inference)
- SGLang
- OpenAI API
- Any OpenAI-compatible endpoint

Designed for simplicity - just point at a server and go.
"""

import os
import time
import random
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ServerType(Enum):
    """Server type hints for common configurations."""
    VLLM = "vllm"
    SGLANG = "sglang"
    OPENAI = "openai"
    CUSTOM = "custom"


@dataclass
class LLMConfig:
    """
    Configuration for OpenAI-compatible LLM server.

    Examples:
        # vLLM local server
        config = LLMConfig.vllm(model="meta-llama/Llama-2-7b-chat-hf")

        # SGLang server
        config = LLMConfig.sglang(port=30000)

        # OpenAI API
        config = LLMConfig.openai(model="gpt-4o")
    """
    base_url: str = "http://localhost:8000/v1"
    model: str = "default"
    api_key: str = "EMPTY"  # vLLM/SGLang don't need real keys
    max_tokens: int = 2000
    temperature: float = 0.7
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 120.0
    server_type: ServerType = ServerType.VLLM

    @classmethod
    def vllm(
        cls,
        model: str = "default",
        host: str = "localhost",
        port: int = 8000,
        **kwargs
    ) -> 'LLMConfig':
        """Create config for vLLM server."""
        return cls(
            base_url=f"http://{host}:{port}/v1",
            model=model,
            api_key="EMPTY",
            server_type=ServerType.VLLM,
            **kwargs
        )

    @classmethod
    def sglang(
        cls,
        model: str = "default",
        host: str = "localhost",
        port: int = 30000,
        **kwargs
    ) -> 'LLMConfig':
        """Create config for SGLang server."""
        return cls(
            base_url=f"http://{host}:{port}/v1",
            model=model,
            api_key="EMPTY",
            server_type=ServerType.SGLANG,
            **kwargs
        )

    @classmethod
    def openai(
        cls,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        **kwargs
    ) -> 'LLMConfig':
        """Create config for OpenAI API."""
        key = api_key or os.getenv('OPENAI_API_KEY', '')
        return cls(
            base_url="https://api.openai.com/v1",
            model=model,
            api_key=key,
            server_type=ServerType.OPENAI,
            **kwargs
        )

    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Create config from environment variables."""
        base_url = os.getenv('LLM_BASE_URL', 'http://localhost:8000/v1')
        model = os.getenv('LLM_MODEL', 'default')
        api_key = os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY', 'EMPTY')
        return cls(base_url=base_url, model=model, api_key=api_key)


@dataclass
class LLMResponse:
    """Response from LLM call."""
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_response: Optional[Any] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """
    OpenAI-compatible LLM client.

    Works with vLLM, SGLang, OpenAI, and any OpenAI-compatible server.

    Example:
        # With vLLM
        client = LLMClient(LLMConfig.vllm(model="llama-2-7b"))
        response = client("Summarize this text...")

        # With SGLang
        client = LLMClient(LLMConfig.sglang())
        response = client.generate("Hello, world!")

        # With OpenAI
        client = LLMClient(LLMConfig.openai(model="gpt-4o"))
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._client = None
        self._usage_lock = threading.Lock()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._call_count = 0

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
        return self._client

    def __call__(self, prompt: str, **kwargs) -> str:
        """Call the LLM and return just the content."""
        response = self.generate(prompt, **kwargs)
        return response.content

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: User message/prompt
            system: Optional system message
            **kwargs: Additional args passed to chat.completions.create

        Returns:
            LLMResponse with content and usage info
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return self.chat(messages, **kwargs)

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """
        Send chat messages to the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional args passed to chat.completions.create

        Returns:
            LLMResponse with content and usage info
        """
        client = self._get_client()
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = client.chat.completions.create(
                    model=kwargs.pop('model', self.config.model),
                    messages=messages,
                    max_tokens=kwargs.pop('max_tokens', self.config.max_tokens),
                    temperature=kwargs.pop('temperature', self.config.temperature),
                    **kwargs
                )

                # Track usage
                with self._usage_lock:
                    self._call_count += 1
                    if response.usage:
                        self._prompt_tokens += response.usage.prompt_tokens
                        self._completion_tokens += response.usage.completion_tokens

                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=response.model,
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                    raw_response=response
                )

            except Exception as e:
                last_error = e
                delay = self.config.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}. Retrying in {delay:.1f}s")
                time.sleep(delay)

        raise RuntimeError(f"LLM call failed after {self.config.max_retries} attempts: {last_error}")

    def get_usage(self) -> Dict[str, int]:
        """Get current token usage."""
        with self._usage_lock:
            return {
                'prompt_tokens': self._prompt_tokens,
                'completion_tokens': self._completion_tokens,
                'total_tokens': self._prompt_tokens + self._completion_tokens,
                'call_count': self._call_count
            }

    def reset_usage(self) -> Dict[str, int]:
        """Get usage and reset counters."""
        with self._usage_lock:
            usage = {
                'prompt_tokens': self._prompt_tokens,
                'completion_tokens': self._completion_tokens,
                'total_tokens': self._prompt_tokens + self._completion_tokens,
                'call_count': self._call_count
            }
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._call_count = 0
            return usage


class MockLLMClient:
    """Mock LLM client for testing without a server."""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        response_fn: Optional[Callable[[str], str]] = None
    ):
        self.config = config or LLMConfig()
        self.response_fn = response_fn or self._default_response
        self.calls: List[str] = []
        self._call_count = 0

    def _default_response(self, prompt: str) -> str:
        """Default mock response: truncate input."""
        if len(prompt) > 100:
            return f"Summary: {prompt[:50]}..."
        return f"Response to: {prompt[:30]}..."

    def __call__(self, prompt: str, **kwargs) -> str:
        """Call the mock LLM."""
        return self.generate(prompt, **kwargs).content

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate mock response."""
        self.calls.append(prompt)
        self._call_count += 1
        content = self.response_fn(prompt)
        return LLMResponse(
            content=content,
            model="mock",
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(content.split())
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Handle chat-style input."""
        prompt = messages[-1]['content'] if messages else ""
        return self.generate(prompt, **kwargs)

    def reset(self) -> None:
        """Reset call history."""
        self.calls = []
        self._call_count = 0

    def get_usage(self) -> Dict[str, int]:
        """Get usage stats."""
        return {'call_count': self._call_count, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}


def create_client(config: Optional[LLMConfig] = None, mock: bool = False) -> LLMClient:
    """
    Create an LLM client.

    Args:
        config: LLM configuration (uses defaults if None)
        mock: If True, return MockLLMClient for testing

    Returns:
        LLMClient or MockLLMClient
    """
    if mock:
        return MockLLMClient(config)
    return LLMClient(config)


def create_summarizer(
    client: Optional[LLMClient] = None,
    system_prompt: Optional[str] = None
) -> Callable[[str, str], str]:
    """
    Create a summarizer function compatible with OPSTreeBuilder.

    Args:
        client: LLM client (creates mock if None)
        system_prompt: Optional system prompt for summarization

    Returns:
        Callable that takes (content, rubric) and returns summary
    """
    if client is None:
        client = MockLLMClient()

    default_system = "You are a precise summarizer. Preserve all information specified in the rubric."

    def summarizer(content: str, rubric: str) -> str:
        prompt = f"""Summarize the following content while preserving information specified in the rubric.

Rubric: {rubric}

Content:
{content}

Summary:"""
        if hasattr(client, 'generate'):
            return client.generate(prompt, system=system_prompt or default_system).content
        return client(prompt)

    return summarizer


# Convenience aliases
def vllm_client(model: str = "default", host: str = "localhost", port: int = 8000, **kwargs) -> LLMClient:
    """Create client for vLLM server."""
    return LLMClient(LLMConfig.vllm(model=model, host=host, port=port, **kwargs))


def sglang_client(model: str = "default", host: str = "localhost", port: int = 30000, **kwargs) -> LLMClient:
    """Create client for SGLang server."""
    return LLMClient(LLMConfig.sglang(model=model, host=host, port=port, **kwargs))


def openai_client(model: str = "gpt-4o", api_key: Optional[str] = None, **kwargs) -> LLMClient:
    """Create client for OpenAI API."""
    return LLMClient(LLMConfig.openai(model=model, api_key=api_key, **kwargs))
