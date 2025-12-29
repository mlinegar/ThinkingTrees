"""
Summarization Strategy Interface.

This module provides a unified async interface for summarization operations,
allowing the same tree-building logic to work with different backends:

1. DSPyStrategy: Wraps DSPy modules (LeafSummarizer, MergeSummarizer) in async
2. BatchedStrategy: Uses AsyncBatchLLMClient for batched inference

Usage:
    # With DSPy modules (for optimization/training)
    strategy = DSPyStrategy(LeafSummarizer(), MergeSummarizer())

    # With batched client (for high-throughput inference)
    async with AsyncBatchLLMClient(url) as client:
        strategy = BatchedStrategy(client)

    # Same tree-building code works with either:
    summary = await strategy.summarize(content, rubric)
    merged = await strategy.merge(left, right, rubric)
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import dspy
    from src.core.batch_processor import AsyncBatchLLMClient, BatchRequest

from src.tasks.prompting import default_merge_prompt, default_summarize_prompt
logger = logging.getLogger(__name__)


class SummarizationStrategy(Protocol):
    """
    Protocol for summarization strategies.

    Both DSPy and batched implementations must provide these async methods.
    """

    async def summarize(self, content: str, rubric: str) -> str:
        """Summarize content according to the rubric."""
        ...

    async def merge(self, left: str, right: str, rubric: str) -> str:
        """Merge two summaries into one."""
        ...


class DSPyStrategy:
    """
    Strategy that wraps DSPy modules in async interface.

    Runs DSPy module calls in a thread pool to avoid blocking the event loop.
    This allows DSPy's synchronous LLM calls to work within an async context.

    Args:
        leaf_module: DSPy module for leaf summarization (content, rubric) -> str
        merge_module: DSPy module for merge summarization (left, right, rubric) -> str
    """

    def __init__(
        self,
        leaf_module: "dspy.Module",
        merge_module: "dspy.Module",
    ):
        self.leaf_module = leaf_module
        self.merge_module = merge_module

    async def summarize(self, content: str, rubric: str) -> str:
        """Summarize content using DSPy leaf module."""
        return await asyncio.to_thread(
            self.leaf_module,
            content=content,
            rubric=rubric
        )

    async def merge(self, left: str, right: str, rubric: str) -> str:
        """Merge summaries using DSPy merge module."""
        return await asyncio.to_thread(
            self.merge_module,
            left_summary=left,
            right_summary=right,
            rubric=rubric
        )


class BatchedStrategy:
    """
    Strategy that uses AsyncBatchLLMClient for batched inference.

    Submits requests to the batch client and awaits responses. The client
    handles batching multiple concurrent requests for optimal GPU utilization.

    Args:
        client: AsyncBatchLLMClient instance (must be started)
        max_tokens: Maximum tokens for summary responses
        summarize_prompt_fn: Function to build summarize prompts
        merge_prompt_fn: Function to build merge prompts
    """

    def __init__(
        self,
        client: "AsyncBatchLLMClient",
        max_tokens: int = 500,
        summarize_prompt_fn=None,
        merge_prompt_fn=None,
    ):
        self.client = client
        self.max_tokens = max_tokens
        self._counter = 0

        # Use default prompt builders if not provided
        if summarize_prompt_fn is None:
            summarize_prompt_fn = default_summarize_prompt
        if merge_prompt_fn is None:
            merge_prompt_fn = default_merge_prompt

        self.summarize_prompt_fn = summarize_prompt_fn
        self.merge_prompt_fn = merge_prompt_fn

    async def summarize(self, content: str, rubric: str) -> str:
        """Summarize content using batched LLM client."""
        from src.core.batch_processor import BatchRequest

        self._counter += 1
        request = BatchRequest(
            request_id=f"strategy_summarize_{self._counter}",
            messages=self.summarize_prompt_fn(content, rubric),
            max_tokens=self.max_tokens,
            request_type="summarize",
        )
        await self.client.submit(request)
        response = await self.client.await_response(request.request_id)
        return response.content if not response.error else ""

    async def merge(self, left: str, right: str, rubric: str) -> str:
        """Merge summaries using batched LLM client."""
        from src.core.batch_processor import BatchRequest

        self._counter += 1
        request = BatchRequest(
            request_id=f"strategy_merge_{self._counter}",
            messages=self.merge_prompt_fn(left, right, rubric),
            max_tokens=self.max_tokens,
            request_type="merge",
        )
        await self.client.submit(request)
        response = await self.client.await_response(request.request_id)
        return response.content if not response.error else ""


# =============================================================================
# Factory Functions
# =============================================================================

def dspy_strategy(
    leaf_module: "dspy.Module",
    merge_module: "dspy.Module",
) -> DSPyStrategy:
    """
    Create a DSPy-based strategy.

    Args:
        leaf_module: DSPy module for leaf summarization
        merge_module: DSPy module for merge summarization

    Returns:
        DSPyStrategy instance
    """
    return DSPyStrategy(leaf_module, merge_module)


def batched_strategy(
    client: "AsyncBatchLLMClient",
    max_tokens: int = 500,
    summarize_prompt_fn=None,
    merge_prompt_fn=None,
) -> BatchedStrategy:
    """
    Create a batched LLM strategy.

    Args:
        client: AsyncBatchLLMClient instance
        max_tokens: Maximum tokens for responses
        summarize_prompt_fn: Custom summarize prompt builder
        merge_prompt_fn: Custom merge prompt builder

    Returns:
        BatchedStrategy instance
    """
    return BatchedStrategy(
        client=client,
        max_tokens=max_tokens,
        summarize_prompt_fn=summarize_prompt_fn,
        merge_prompt_fn=merge_prompt_fn,
    )
