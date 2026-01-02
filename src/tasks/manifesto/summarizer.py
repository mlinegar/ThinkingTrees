"""
DSPy-based Summarization Modules for RILE Preservation.

This module provides optimizable summarization modules that can be trained
using DSPy's GEPA optimizer to maximize RILE information preservation.

The key difference from the hardcoded prompts in batched_pipeline.py:
- These modules can be optimized with DSPy
- Prompts/instructions can evolve through training
- Demonstrations can be learned from training data

Usage:
    from src.tasks.manifesto import LeafSummarizer, MergeSummarizer

    # Create modules
    leaf_summarizer = LeafSummarizer()
    merge_summarizer = MergeSummarizer()

    # Use directly
    summary = leaf_summarizer(content="...", rubric="...")

    # Or optimize with DSPy GEPA
    optimized = gepa.compile(leaf_summarizer, trainset=trainset)
"""

import dspy
from dataclasses import dataclass
from typing import Optional

from src.core.signatures import RecursiveSummary


# =============================================================================
# RILE-Specific Signature (extends RecursiveSummary with RILE focus)
# =============================================================================

class RILELeafSummary(dspy.Signature):
    """
    Summarize political text while preserving left-right (RILE) positioning information.

    This signature is specifically designed for leaf-level summarization of
    political manifesto chunks. The summary must preserve all information
    relevant to determining the document's position on the left-right spectrum.
    """
    rubric: str = dspy.InputField(
        desc="Information preservation criteria specifying what political indicators to preserve"
    )
    content: str = dspy.InputField(
        desc="Raw political text chunk to summarize"
    )
    summary: str = dspy.OutputField(
        desc="Concise summary that preserves all RILE-relevant political positioning information"
    )


class RILEMergeSummary(dspy.Signature):
    """
    Merge two summaries while preserving combined RILE positioning information.

    This signature is for internal node summarization - combining child summaries
    into a parent summary while ensuring no political positioning information is lost.
    """
    rubric: str = dspy.InputField(
        desc="Information preservation criteria specifying what political indicators to preserve"
    )
    left_summary: str = dspy.InputField(
        desc="First summary to merge"
    )
    right_summary: str = dspy.InputField(
        desc="Second summary to merge"
    )
    merged_summary: str = dspy.OutputField(
        desc="Combined summary preserving all RILE-relevant information from both inputs"
    )


# =============================================================================
# Optimizable Summarizer Modules
# =============================================================================

class LeafSummarizer(dspy.Module):
    """
    Optimizable leaf summarization module for RILE preservation.

    This module can be trained with DSPy's optimizers (GEPA, MIPROv2) to learn
    the best instructions and demonstrations for preserving political
    positioning information during summarization.

    Example:
        summarizer = LeafSummarizer()

        # Direct use
        summary = summarizer(content="The party supports...", rubric=RILE_RUBRIC)

        # With GEPA optimization
        metric = create_summarization_metric(oracle_classifier)
        optimizer = dspy.GEPA(metric=metric, auto='light')
        optimized_summarizer = optimizer.compile(summarizer, trainset=trainset)
    """

    def __init__(self, use_cot: bool = True):
        """
        Initialize the leaf summarizer.

        Args:
            use_cot: Whether to use Chain-of-Thought reasoning (recommended for
                     better preservation of nuanced political content)
        """
        super().__init__()
        if use_cot:
            self.summarize = dspy.ChainOfThought(RILELeafSummary)
        else:
            self.summarize = dspy.Predict(RILELeafSummary)

    def forward(self, content: str, rubric: str) -> str:
        """
        Generate a RILE-preserving summary of the content.

        Args:
            content: Raw political text chunk to summarize
            rubric: Information preservation criteria

        Returns:
            Summary string preserving RILE-relevant information
        """
        result = self.summarize(content=content, rubric=rubric)
        return result.summary


class MergeSummarizer(dspy.Module):
    """
    Optimizable merge summarization module for RILE preservation.

    This module combines two summaries while preserving all political
    positioning information from both. Can be trained separately from
    leaf summarization since the task is different.

    Example:
        merger = MergeSummarizer()

        # Direct use
        merged = merger(
            left_summary="Summary A...",
            right_summary="Summary B...",
            rubric=RILE_RUBRIC
        )

        # With optimization
        optimized_merger = optimizer.compile(merger, trainset=merge_trainset)
    """

    def __init__(self, use_cot: bool = True):
        """
        Initialize the merge summarizer.

        Args:
            use_cot: Whether to use Chain-of-Thought reasoning
        """
        super().__init__()
        if use_cot:
            self.merge = dspy.ChainOfThought(RILEMergeSummary)
        else:
            self.merge = dspy.Predict(RILEMergeSummary)

    def forward(self, left_summary: str, right_summary: str, rubric: str) -> str:
        """
        Merge two summaries while preserving RILE information.

        Args:
            left_summary: First summary to merge
            right_summary: Second summary to merge
            rubric: Information preservation criteria

        Returns:
            Merged summary string
        """
        result = self.merge(
            left_summary=left_summary,
            right_summary=right_summary,
            rubric=rubric
        )
        return result.merged_summary


# =============================================================================
# Generic Summarizer (Uses RecursiveSummary from core)
# =============================================================================

class GenericSummarizer(dspy.Module):
    """
    Generic summarizer using the core RecursiveSummary signature.

    This provides compatibility with the existing OPS infrastructure while
    still being optimizable through DSPy.
    """

    def __init__(self, use_cot: bool = True):
        super().__init__()
        if use_cot:
            self.summarize = dspy.ChainOfThought(RecursiveSummary)
        else:
            self.summarize = dspy.Predict(RecursiveSummary)

    def forward(self, content: str, rubric: str) -> str:
        """Generate summary using the generic RecursiveSummary signature."""
        result = self.summarize(content=content, rubric=rubric)
        return result.summary


class GenericMerger(dspy.Module):
    """
    Generic merger using RecursiveSummary for merge operations.

    Combines summaries by concatenating them as content and re-summarizing.
    """

    def __init__(self, use_cot: bool = True):
        super().__init__()
        if use_cot:
            self.merge = dspy.ChainOfThought(RecursiveSummary)
        else:
            self.merge = dspy.Predict(RecursiveSummary)

    def forward(self, left_summary: str, right_summary: str, rubric: str) -> str:
        """Merge summaries by re-summarizing their concatenation."""
        combined = f"PART 1:\n{left_summary}\n\nPART 2:\n{right_summary}"
        result = self.merge(content=combined, rubric=rubric)
        return result.summary


# =============================================================================
# Factory Functions
# =============================================================================

def create_summarizers(
    use_rile_specific: bool = True,
    use_cot: bool = True,
) -> tuple:
    """
    Create leaf and merge summarizer modules.

    Args:
        use_rile_specific: Use RILE-specific signatures (recommended for manifesto work)
        use_cot: Use Chain-of-Thought reasoning

    Returns:
        Tuple of (leaf_summarizer, merge_summarizer)
    """
    if use_rile_specific:
        return LeafSummarizer(use_cot=use_cot), MergeSummarizer(use_cot=use_cot)
    else:
        return GenericSummarizer(use_cot=use_cot), GenericMerger(use_cot=use_cot)


# =============================================================================
# Dataclass for Summarization Results
# =============================================================================

@dataclass
class SummarizationResult:
    """Result from a summarization operation."""
    summary: str
    input_length: int
    output_length: int
    compression_ratio: float

    @classmethod
    def from_summary(cls, original: str, summary: str) -> "SummarizationResult":
        """Create result from original text and summary."""
        return cls(
            summary=summary,
            input_length=len(original),
            output_length=len(summary),
            compression_ratio=len(summary) / max(len(original), 1),
        )
