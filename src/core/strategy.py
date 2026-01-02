"""
Summarization Strategy Interface.

This module provides a unified async interface for summarization operations,
allowing the same tree-building logic to work with different backends:

1. BatchedStrategy: Uses AsyncBatchLLMClient for batched inference (foundation)
2. DSPyStrategy: Wraps DSPy modules, uses batching internally
3. CallableStrategy: Wraps a sync callable with async + temperature support
4. TournamentStrategy: Wraps any strategy with tournament selection for learning

Architecture:
    Batching is ALWAYS the foundation. DSPy is an optional layer for optimization.
    All strategies support temperature and candidate generation via batching.

Usage:
    # Batched inference (foundation)
    async with AsyncBatchLLMClient(url) as client:
        strategy = BatchedStrategy(client)

    # With DSPy for optimization (uses batching internally)
    strategy = DSPyStrategy(LeafSummarizer(), MergeSummarizer())

    # With tournament selection (for learning with preference collection)
    strategy = TournamentStrategy(
        base=BatchedStrategy(client),
        judge=GenRMJudge(genrm_url),
    )

    # Same tree-building code works with all:
    summary = await strategy.summarize(content, rubric)
    merged = await strategy.merge(left, right, rubric)

    # Generate diverse candidates (all strategies support this)
    candidates = await strategy.generate_candidates(content, rubric, k=4)
"""

import asyncio
import logging
import random
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Protocol, TYPE_CHECKING, Literal, Union, Callable

if TYPE_CHECKING:
    import dspy
    from src.core.batch_processor import AsyncBatchLLMClient, BatchRequest
    from src.ops_engine.training_framework.genrm_preference import GenRMJudge
    from src.ops_engine.training_framework.genrm_dspy import GenRMComparisonModule
    from src.ops_engine.training_framework.preference import PreferencePair

from src.core.prompting import default_merge_prompt, default_summarize_prompt
logger = logging.getLogger(__name__)

# Context for routing tournament preferences (e.g., batch doc IDs).
tournament_doc_id: ContextVar[Optional[str]] = ContextVar("tournament_doc_id", default=None)


def _filter_valid_candidates(results: List[Any], operation: str = "generation") -> List[str]:
    """Filter async gather results to valid non-empty string candidates.

    Args:
        results: Results from asyncio.gather with return_exceptions=True
        operation: Description for logging (e.g., "Candidate generation", "Merge candidate")

    Returns:
        List of valid non-empty string candidates
    """
    candidates = []
    for result in results:
        if isinstance(result, str) and result.strip():
            candidates.append(result)
        elif isinstance(result, Exception):
            logger.debug(f"{operation} failed: {result}")
    return candidates


# =============================================================================
# Strategy Protocol
# =============================================================================

class SummarizationStrategy(Protocol):
    """
    Protocol for summarization strategies.

    All strategies support:
    - temperature parameter for controlling diversity
    - candidate generation via batching at high temperature
    """

    async def summarize(
        self, content: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """Summarize content according to the rubric."""
        ...

    async def merge(
        self, left: str, right: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """Merge two summaries into one."""
        ...

    async def generate_candidates(
        self, content: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """
        Generate k diverse candidate summaries.

        Uses batching at high temperature for diversity.
        """
        ...

    async def generate_merge_candidates(
        self, left: str, right: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """
        Generate k diverse merge candidates.

        Uses batching at high temperature for diversity.
        """
        ...


# =============================================================================
# Batched Strategy (Foundation)
# =============================================================================

class BatchedStrategy:
    """
    Strategy using AsyncBatchLLMClient for batched inference.

    This is the FOUNDATION strategy - all other strategies build on batching.
    Submits requests to the batch client which handles pooling for optimal
    GPU utilization.

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

    async def summarize(
        self, content: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """Summarize content using batched LLM client."""
        from src.core.batch_processor import BatchRequest

        self._counter += 1
        request = BatchRequest(
            request_id=f"strategy_summarize_{self._counter}",
            messages=self.summarize_prompt_fn(content, rubric),
            max_tokens=self.max_tokens,
            request_type="summarize",
            temperature=temperature,
        )
        await self.client.submit(request)
        response = await self.client.await_response(request.request_id)
        return response.content if not response.error else ""

    async def merge(
        self, left: str, right: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """Merge summaries using batched LLM client."""
        from src.core.batch_processor import BatchRequest

        self._counter += 1
        request = BatchRequest(
            request_id=f"strategy_merge_{self._counter}",
            messages=self.merge_prompt_fn(left, right, rubric),
            max_tokens=self.max_tokens,
            request_type="merge",
            temperature=temperature,
        )
        await self.client.submit(request)
        response = await self.client.await_response(request.request_id)
        return response.content if not response.error else ""

    async def _generate_candidates_impl(
        self,
        messages: List[Dict[str, str]],
        request_type: str,
        k: int,
        temperature: float,
    ) -> List[str]:
        """Common implementation for candidate generation.

        Args:
            messages: Prompt messages to send for each candidate
            request_type: Type identifier for requests (e.g., "candidate", "merge_candidate")
            k: Number of candidates to generate
            temperature: Sampling temperature for diversity

        Returns:
            List of generated candidate strings
        """
        from src.core.batch_processor import BatchRequest

        # Submit k requests in parallel
        requests = []
        for _ in range(k):
            self._counter += 1
            request = BatchRequest(
                request_id=f"strategy_{request_type}_{self._counter}",
                messages=messages,
                max_tokens=self.max_tokens,
                request_type=request_type,
                temperature=temperature,
            )
            requests.append(request)
            await self.client.submit(request)

        # Await all responses
        candidates = []
        for request in requests:
            response = await self.client.await_response(request.request_id)
            if response.content and not response.error:
                candidates.append(response.content)

        return candidates

    async def generate_candidates(
        self, content: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """Generate k diverse candidates via batched requests at high temperature."""
        return await self._generate_candidates_impl(
            messages=self.summarize_prompt_fn(content, rubric),
            request_type="candidate",
            k=k,
            temperature=temperature,
        )

    async def generate_merge_candidates(
        self, left: str, right: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """Generate k diverse merge candidates via batched requests at high temperature."""
        return await self._generate_candidates_impl(
            messages=self.merge_prompt_fn(left, right, rubric),
            request_type="merge_candidate",
            k=k,
            temperature=temperature,
        )


# =============================================================================
# DSPy Strategy (wraps DSPy modules, can use batching internally)
# =============================================================================

class DSPyStrategy:
    """
    Strategy that wraps DSPy modules in async interface.

    Runs DSPy module calls in a thread pool to avoid blocking the event loop.
    For candidate generation, uses parallel calls at high temperature.

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

    async def summarize(
        self, content: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """Summarize content using DSPy leaf module."""
        return await asyncio.to_thread(
            self._call_with_temp,
            self.leaf_module,
            temperature,
            content=content,
            rubric=rubric,
        )

    async def merge(
        self, left: str, right: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """Merge summaries using DSPy merge module."""
        return await asyncio.to_thread(
            self._call_with_temp,
            self.merge_module,
            temperature,
            left_summary=left,
            right_summary=right,
            rubric=rubric,
        )

    async def generate_candidates(
        self, content: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """
        Generate k diverse candidates using parallel DSPy calls at high temperature.
        """
        # Launch k calls in parallel
        tasks = [
            asyncio.to_thread(
                self._call_with_temp,
                self.leaf_module,
                temperature,
                content=content,
                rubric=rubric,
            )
            for _ in range(k)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return _filter_valid_candidates(results, "Candidate generation")

    async def generate_merge_candidates(
        self, left: str, right: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """
        Generate k diverse merge candidates using parallel DSPy calls at high temperature.
        """
        tasks = [
            asyncio.to_thread(
                self._call_with_temp,
                self.merge_module,
                temperature,
                left_summary=left,
                right_summary=right,
                rubric=rubric,
            )
            for _ in range(k)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return _filter_valid_candidates(results, "Merge candidate generation")

    def _call_with_temp(self, module, temperature: float, **kwargs) -> str:
        """Call DSPy module with specific temperature (sync, runs in thread)."""
        import dspy

        current_lm = dspy.settings.lm
        with dspy.context(lm=current_lm.copy(temperature=temperature)):
            result = module(**kwargs)
            return getattr(result, 'summary', str(result))


# =============================================================================
# Callable Strategy (wraps a sync callable, temperature-aware if DSPy is configured)
# =============================================================================

class CallableStrategy:
    """
    Strategy that wraps a sync callable with the SummarizationStrategy interface.

    This is useful for integrating sync summarizers (e.g., DSPy modules) into
    async tree-building while preserving temperature control when DSPy is active.
    """

    def __init__(
        self,
        summarizer: Callable[..., Any],
        merge_fn: Optional[Callable[..., Any]] = None,
    ):
        """
        Initialize callable strategy.

        Args:
            summarizer: Sync function/content-based module (content, rubric) -> str
            merge_fn: Optional sync function for merges (left_summary, right_summary, rubric) -> str
        """
        self.summarizer = summarizer
        self.merge_fn = merge_fn

    async def summarize(
        self, content: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """Summarize content using the wrapped callable."""
        return await asyncio.to_thread(
            self._call_with_temp,
            self.summarizer,
            temperature,
            content=content,
            rubric=rubric,
        )

    async def merge(
        self, left: str, right: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """Merge summaries using the wrapped callable."""
        if self.merge_fn is not None:
            return await asyncio.to_thread(
                self._call_with_temp,
                self.merge_fn,
                temperature,
                left_summary=left,
                right_summary=right,
                rubric=rubric,
            )

        combined = f"{left}\n\n{right}"
        return await asyncio.to_thread(
            self._call_with_temp,
            self.summarizer,
            temperature,
            content=combined,
            rubric=rubric,
        )

    async def generate_candidates(
        self, content: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """Generate k candidates using parallel calls at a fixed temperature."""
        tasks = [
            asyncio.to_thread(
                self._call_with_temp,
                self.summarizer,
                temperature,
                content=content,
                rubric=rubric,
            )
            for _ in range(k)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return _filter_valid_candidates(results, "Candidate generation")

    async def generate_merge_candidates(
        self, left: str, right: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """Generate k merge candidates using parallel calls at a fixed temperature."""
        if self.merge_fn is not None:
            tasks = [
                asyncio.to_thread(
                    self._call_with_temp,
                    self.merge_fn,
                    temperature,
                    left_summary=left,
                    right_summary=right,
                    rubric=rubric,
                )
                for _ in range(k)
            ]
        else:
            combined = f"{left}\n\n{right}"
            tasks = [
                asyncio.to_thread(
                    self._call_with_temp,
                    self.summarizer,
                    temperature,
                    content=combined,
                    rubric=rubric,
                )
                for _ in range(k)
            ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return _filter_valid_candidates(results, "Merge candidate generation")

    def _call_with_temp(self, fn, temperature: float, **kwargs) -> str:
        """Call wrapped function with DSPy temperature context when available."""
        try:
            import dspy
        except Exception:
            result = fn(**kwargs)
            return getattr(result, 'summary', str(result))

        current_lm = getattr(dspy.settings, 'lm', None)
        if current_lm is None:
            result = fn(**kwargs)
            return getattr(result, 'summary', str(result))

        with dspy.context(lm=current_lm.copy(temperature=temperature)):
            result = fn(**kwargs)
            return getattr(result, 'summary', str(result))


# =============================================================================
# Tournament Strategy (Wrapper for Learning Mode)
# =============================================================================

@dataclass
class TournamentConfig:
    """Configuration for tournament-based candidate selection."""
    k: int = 4  # Number of candidates to generate
    temperature: float = 0.9  # Temperature for candidate generation


class TournamentStrategy:
    """
    Wraps any SummarizationStrategy with tournament selection.

    This strategy generates multiple candidate summaries using the base strategy's
    generate_candidates() method and uses a GenRM judge to select the best one
    via pairwise tournament. Preference pairs are collected as a FREE byproduct.

    Usage:
        # Wrap any base strategy
        base = BatchedStrategy(client)
        strategy = TournamentStrategy(base, judge=GenRMJudge(...))

        # Use like any other strategy - tournament happens transparently
        summary = await strategy.summarize(content, rubric)

        # Get collected preferences (free byproduct!)
        preferences = strategy.get_preferences()

    The tournament wrapper is transparent to TreeBuilder - it doesn't know
    or care that tournament selection is happening internally.
    """

    def __init__(
        self,
        base: SummarizationStrategy,
        judge: Union["GenRMJudge", "GenRMComparisonModule"],
        config: Optional[TournamentConfig] = None,
    ):
        """
        Initialize tournament strategy.

        Args:
            base: Base summarization strategy to wrap
            judge: GenRMJudge or GenRMComparisonModule for pairwise comparison.
                   GenRMComparisonModule can use optimizable DSPy prompts when
                   initialized with use_dspy_predictor=True (for tournament of tournaments).
            config: Tournament configuration (k candidates, temperature)
        """
        self.base = base
        self.judge = judge
        self.config = config or TournamentConfig()
        self._preferences: List["PreferencePair"] = []
        self._segment_counter = 0

    async def summarize(
        self, content: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """
        Summarize with tournament selection.

        Generates k candidates via base strategy, runs tournament, returns winner.
        Collects preferences as byproduct.

        Note: temperature param is ignored - we use config.temperature for candidates.
        """
        self._segment_counter += 1
        segment_id = f"leaf_{self._segment_counter}"

        # Generate k candidates using base strategy
        candidates = await self.base.generate_candidates(
            content, rubric, k=self.config.k, temperature=self.config.temperature
        )

        if len(candidates) < 2:
            return candidates[0] if candidates else ""

        # Run tournament and collect preferences
        winner, prefs = await self._run_tournament(
            candidates, content, rubric, segment_id, law_type="sufficiency"
        )
        self._preferences.extend(prefs)

        return winner

    async def merge(
        self, left: str, right: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """
        Merge with tournament selection.

        Generates k merge candidates via base strategy, runs tournament, returns winner.
        Collects preferences as byproduct.
        """
        self._segment_counter += 1
        segment_id = f"merge_{self._segment_counter}"

        # Generate k candidates using base strategy
        candidates = await self.base.generate_merge_candidates(
            left, right, rubric, k=self.config.k, temperature=self.config.temperature
        )

        if len(candidates) < 2:
            return candidates[0] if candidates else ""

        # For merge operations, the "original" is the concatenated child summaries
        original_text = f"{left}\n\n{right}"

        # Run tournament and collect preferences
        winner, prefs = await self._run_tournament(
            candidates, original_text, rubric, segment_id, law_type="merge"
        )
        self._preferences.extend(prefs)

        return winner

    async def generate_candidates(
        self, content: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """Delegate to base strategy."""
        return await self.base.generate_candidates(content, rubric, k, temperature)

    async def generate_merge_candidates(
        self, left: str, right: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """Delegate to base strategy."""
        return await self.base.generate_merge_candidates(left, right, rubric, k, temperature)

    async def _run_tournament(
        self,
        candidates: List[str],
        original_text: str,
        rubric: str,
        segment_id: str,
        law_type: str = "sufficiency",
    ) -> tuple[str, List["PreferencePair"]]:
        """
        Run elimination tournament and collect preferences.

        Supports both GenRMJudge (uses .compare()) and GenRMComparisonModule
        (uses .forward() for DSPy compatibility).

        Returns (winner, list of PreferencePair).
        """
        from src.ops_engine.training_framework.preference import PreferencePair

        if len(candidates) == 0:
            raise ValueError("No candidates provided")
        if len(candidates) == 1:
            return candidates[0], []

        remaining = candidates.copy()
        preferences: List["PreferencePair"] = []
        round_num = 0

        doc_id = tournament_doc_id.get()
        segment_tag = f"{doc_id}:{segment_id}" if doc_id is not None else segment_id

        # Determine judge type and model name
        is_dspy_module = hasattr(self.judge, 'forward') and hasattr(self.judge, 'use_dspy_predictor')
        if is_dspy_module:
            if getattr(self.judge, 'use_dspy_prompt', False):
                judge_model = "genrm-prompt-tuned"
            elif self.judge.use_dspy_predictor:
                judge_model = "dspy-optimizable"
            else:
                judge_model = "genrm-via-dspy"
        else:
            judge_model = "qwen3-nemotron-genrm"

        while len(remaining) > 1:
            # Collect all matches for this round
            matches = []  # List of (idx_a, idx_b, summary_a, summary_b)
            for i in range(0, len(remaining), 2):
                if i + 1 < len(remaining):
                    matches.append((i, i + 1, remaining[i], remaining[i + 1]))

            if not matches:
                break

            # Batch execute all comparisons for this round
            if is_dspy_module:
                # DSPy modules: use asyncio.gather with to_thread
                tasks = [
                    asyncio.to_thread(
                        self.judge.forward,
                        context=rubric,
                        original_text=original_text,
                        summary_a=summary_a,
                        summary_b=summary_b,
                        law_type=law_type,
                    )
                    for _, _, summary_a, summary_b in matches
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            elif hasattr(self.judge, 'compare_batch_async'):
                # GenRMJudge with batch support: use compare_batch_async
                comparisons = [
                    (rubric, original_text, summary_a, summary_b, law_type, None)
                    for _, _, summary_a, summary_b in matches
                ]
                results = await self.judge.compare_batch_async(comparisons)
            elif hasattr(self.judge, 'compare_async'):
                # GenRMJudge without batch: use asyncio.gather on compare_async
                tasks = [
                    self.judge.compare_async(
                        context=rubric,
                        original_text=original_text,
                        summary_a=summary_a,
                        summary_b=summary_b,
                        law_type=law_type,
                    )
                    for _, _, summary_a, summary_b in matches
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Fallback: sync compare in threads
                tasks = [
                    asyncio.to_thread(
                        self.judge.compare,
                        context=rubric,
                        original_text=original_text,
                        summary_a=summary_a,
                        summary_b=summary_b,
                        law_type=law_type,
                    )
                    for _, _, summary_a, summary_b in matches
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process all results for this round
            next_round = []
            for match_num, ((idx_a, idx_b, summary_a, summary_b), result) in enumerate(zip(matches, results)):
                # Handle exceptions from gather
                if isinstance(result, Exception):
                    logger.warning(f"Tournament match failed: {result}, treating as tie")
                    preferred = "tie"
                    result = None  # No result object available
                elif is_dspy_module:
                    # DSPy Prediction uses .preference instead of .preferred
                    preferred = getattr(result, 'preference', getattr(result, 'preferred', 'tie'))
                else:
                    preferred = result.preferred

                # Capture preference pair (FREE - no extra cost!)
                pair = PreferencePair(
                    pair_id=f"tournament_{segment_tag}_r{round_num}_m{match_num}",
                    source_example_id=segment_tag,
                    original_text=original_text,  # Store full text - truncation corrupts training data
                    rubric=rubric,
                    reference_score=None,
                    summary_a=summary_a,
                    summary_b=summary_b,
                    preferred=preferred,
                    reasoning=getattr(result, 'reasoning', "") if result else "",
                    confidence=getattr(result, 'confidence', 0.5) if result else 0.5,
                    law_type=law_type,
                    score_estimate_a=getattr(result, 'helpfulness_a', None) if result else None,
                    score_estimate_b=getattr(result, 'helpfulness_b', None) if result else None,
                    judge_model=judge_model,
                )
                preferences.append(pair)

                # Handle tie with random selection to avoid position bias
                if preferred == "A":
                    winner = summary_a
                elif preferred == "B":
                    winner = summary_b
                else:  # tie
                    winner = summary_a if random.random() < 0.5 else summary_b
                next_round.append(winner)

            # Handle odd candidate (advances without playing)
            if len(remaining) % 2 == 1:
                next_round.append(remaining[-1])

            remaining = next_round
            round_num += 1

        return remaining[0], preferences

    def get_preferences(self) -> List["PreferencePair"]:
        """Get all collected preference pairs."""
        return self._preferences

    def reset_preferences(self) -> None:
        """Reset collected preferences (e.g., between documents)."""
        self._preferences = []

    def get_preference_count(self) -> int:
        """Get number of collected preferences."""
        return len(self._preferences)


# =============================================================================
# Strategy Registry
# =============================================================================

_STRATEGY_REGISTRY: Dict[str, type] = {}


def register_strategy(name: str):
    """
    Decorator to register a strategy class.

    Args:
        name: Name to register the strategy under

    Example:
        @register_strategy("custom")
        class CustomStrategy:
            ...
    """
    def decorator(cls):
        _STRATEGY_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_strategy(name: str, **kwargs) -> SummarizationStrategy:
    """
    Get a strategy by name from the registry.

    Args:
        name: Strategy name ("batched", "dspy", "callable", "tournament")
        **kwargs: Arguments passed to strategy constructor

    Returns:
        Configured strategy instance

    Raises:
        ValueError: If strategy name is not registered

    Example:
        strategy = get_strategy("batched", client=my_client)
        strategy = get_strategy("tournament", base=base_strategy, judge=my_judge)
    """
    name_lower = name.lower()
    if name_lower not in _STRATEGY_REGISTRY:
        available = list(_STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy: '{name}'. Available: {available}")

    return _STRATEGY_REGISTRY[name_lower](**kwargs)


def list_strategies() -> List[str]:
    """Return list of registered strategy names."""
    return list(_STRATEGY_REGISTRY.keys())


# Register built-in strategies
register_strategy("batched")(BatchedStrategy)
register_strategy("dspy")(DSPyStrategy)
register_strategy("callable")(CallableStrategy)
register_strategy("tournament")(TournamentStrategy)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Protocol
    "SummarizationStrategy",
    # Registry
    "get_strategy",
    "list_strategies",
    "register_strategy",
    # Implementations
    "BatchedStrategy",
    "DSPyStrategy",
    "CallableStrategy",
    "TournamentStrategy",
    "TournamentConfig",
    # Context
    "tournament_doc_id",
]
