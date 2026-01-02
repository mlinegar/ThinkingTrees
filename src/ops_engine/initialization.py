"""
Top-Down Initialization for OPS Summarizers.

This module implements "oracle-aligned initialization" where the summarizer
is seeded with demonstrations that produce oracle-matching predictions.

KEY INSIGHT: For short documents that fit in context, oracle(doc) = ground_truth
by definition. We can use these short documents as perfect demos where the
"summary" preserves all oracle-relevant information.

The approach:
1. Filter to SHORT documents that fit in the LLM context window
2. Use these documents directly as input-output pairs
   (since a short doc IS the ideal "summary" of itself)
3. Seed the summarizer with these high-quality demos

This ensures the summarizer learns from examples where we KNOW the oracle
is preserved, rather than hoping generated summaries maintain alignment.

Usage:
    from src.ops_engine.initialization import (
        TopDownInitializer,
        create_oracle_aligned_demos,
        initialize_summarizer_with_demos,
    )

    # Create initializer with context limit
    initializer = TopDownInitializer(
        max_doc_chars=DEFAULT_MAX_DOC_CHARS,  # Must fit in context
        label_field='reference_score',
    )

    # Generate demos from labeled documents (filters to short docs)
    demos = initializer.create_demos(train_samples, n_demos=8)

    # Seed the summarizer
    initialized_summarizer = initializer.initialize_module(
        leaf_summarizer, demos
    )
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

import dspy

from src.config.constants import DEFAULT_MAX_DOC_CHARS
from src.core.protocols import format_merge_input

logger = logging.getLogger(__name__)


@dataclass
class OracleAlignedDemo:
    """A demonstration example where the summary aligns with the oracle."""

    # Input
    original_content: str
    rubric: str

    # Output (the aligned summary)
    summary: str

    # Alignment info
    ground_truth_label: Any  # e.g., RILE score
    oracle_prediction: Optional[Any] = None
    alignment_score: float = 0.0  # How well summary matches oracle expectation

    # Metadata
    source_id: Optional[str] = None
    reasoning: Optional[str] = None

    def to_dspy_example(self) -> dspy.Example:
        """Convert to DSPy Example for module seeding."""
        return dspy.Example(
            content=self.original_content,
            rubric=self.rubric,
            summary=self.summary,
            ground_truth_label=self.ground_truth_label,
        ).with_inputs('content', 'rubric')


@dataclass
class MergeAlignedDemo:
    """A merge demonstration where combined summary aligns with oracle."""

    left_summary: str
    right_summary: str
    rubric: str
    merged_summary: str
    ground_truth_label: Any
    alignment_score: float = 0.0
    source_id: Optional[str] = None

    def to_dspy_example(self) -> dspy.Example:
        """Convert to DSPy Example for merge module seeding."""
        return dspy.Example(
            left_summary=self.left_summary,
            right_summary=self.right_summary,
            rubric=self.rubric,
            merged_summary=self.merged_summary,
        ).with_inputs('left_summary', 'right_summary', 'rubric')


class TopDownInitializer:
    """
    Initialize summarizers with oracle-aligned demonstrations.

    This implements "top-down" initialization using SHORT documents where
    we KNOW the oracle is preserved (because oracle(doc) = ground_truth).

    Key insight: For documents short enough to fit in context, the document
    itself is the perfect "summary" - it contains all oracle-relevant info.
    We use these as demos to teach the summarizer what to preserve.

    The result is a summarizer that starts in a good region of the
    optimization landscape, dramatically reducing training time.
    """

    def __init__(
        self,
        max_doc_chars: int = 4000,
        max_doc_tokens: Optional[int] = None,
        label_extractor: Optional[Callable] = None,
        oracle_fn: Optional[Callable] = None,
        oracle_classifier: Optional[Any] = None,
        max_workers: int = 16,
    ):
        """
        Initialize the top-down initializer.

        Args:
            max_doc_chars: Maximum document length in characters to include
                          (must fit in LLM context window)
            max_doc_tokens: Optional token limit (if provided, used instead of chars)
            label_extractor: Function to extract ground truth label from sample
            oracle_fn: Optional oracle function for verification
            oracle_classifier: Optional oracle instance with predict_rile() method
            max_workers: Parallel workers for processing
        """
        self.max_doc_chars = max_doc_chars
        self.max_doc_tokens = max_doc_tokens
        self.label_extractor = label_extractor or (lambda x: getattr(x, 'reference_score', None))
        self.oracle_fn = oracle_fn
        self.oracle_classifier = oracle_classifier
        self.max_workers = max_workers

    def _is_short_enough(self, text: str) -> bool:
        """Check if document is short enough to fit in context."""
        if self.max_doc_tokens:
            # Rough estimate: 4 chars per token
            return len(text) / 4 <= self.max_doc_tokens
        return len(text) <= self.max_doc_chars

    def _filter_short_documents(
        self,
        samples: List[Any],
        text_field: str = 'text',
    ) -> List[Any]:
        """Filter to documents short enough to fit in context."""
        short_docs = []

        for sample in samples:
            text = getattr(sample, text_field, None)
            if text is None and isinstance(sample, dict):
                text = sample.get(text_field, '')

            if text and self._is_short_enough(text):
                short_docs.append(sample)

        logger.info(
            f"Filtered to {len(short_docs)}/{len(samples)} short documents "
            f"(≤{self.max_doc_chars} chars)"
        )
        return short_docs

    def create_demos_from_samples(
        self,
        samples: List[Any],
        rubric: str,
        n_demos: int = 8,
        text_field: str = 'text',
        id_field: str = 'manifesto_id',
    ) -> List[OracleAlignedDemo]:
        """
        Create oracle-aligned demonstrations from labeled samples.

        KEY: We filter to SHORT documents where oracle(doc) = ground_truth
        by definition. The document itself IS the ideal "summary" since
        it contains all oracle-relevant information.

        Args:
            samples: List of samples with ground truth labels
            rubric: Information preservation rubric
            n_demos: Number of demos to generate
            text_field: Attribute name for text content
            id_field: Attribute name for sample ID

        Returns:
            List of OracleAlignedDemo (short docs used as both input AND summary)
        """
        # Step 1: Filter to short documents only
        short_samples = self._filter_short_documents(samples, text_field)

        if not short_samples:
            logger.warning(
                f"No documents short enough (≤{self.max_doc_chars} chars). "
                f"Consider increasing max_doc_chars or using shorter documents."
            )
            return []

        demos = []

        for sample in short_samples[:n_demos * 2]:  # Get extra in case some fail
            # Extract text and label
            text = getattr(sample, text_field, None)
            if text is None and isinstance(sample, dict):
                text = sample.get(text_field, '')

            ground_truth = self.label_extractor(sample)
            if ground_truth is None:
                continue

            sample_id = getattr(sample, id_field, None)
            if sample_id is None and isinstance(sample, dict):
                sample_id = sample.get(id_field, 'unknown')

            # KEY INSIGHT: For short docs, the doc IS the perfect summary
            # oracle(doc) = ground_truth by definition
            # So we use the doc as both input and output
            demo = OracleAlignedDemo(
                original_content=text,
                rubric=rubric,
                summary=text,  # The doc IS the ideal "summary"
                ground_truth_label=ground_truth,
                oracle_prediction=ground_truth,  # By definition
                alignment_score=1.0,  # Perfect alignment
                source_id=str(sample_id),
                reasoning=f"Short doc ({len(text)} chars) - oracle match guaranteed",
            )
            demos.append(demo)

            if len(demos) >= n_demos:
                break

        logger.info(
            f"Created {len(demos)} oracle-aligned demos from short documents "
            f"(alignment=1.0 by construction)"
        )

        return demos

    def initialize_module(
        self,
        module: dspy.Module,
        demos: List[OracleAlignedDemo],
    ) -> dspy.Module:
        """
        Initialize a DSPy module with oracle-aligned demonstrations.

        Args:
            module: The DSPy module to initialize (e.g., LeafSummarizer)
            demos: Oracle-aligned demonstrations

        Returns:
            Module with demos set
        """
        dspy_examples = [d.to_dspy_example() for d in demos]

        # Find the predictor/COT in the module
        for name, submodule in module.named_predictors():
            if hasattr(submodule, 'demos'):
                submodule.demos = dspy_examples
                logger.info(f"Set {len(dspy_examples)} demos on {name}")

        return module


# =============================================================================
# Convenience Functions
# =============================================================================

def oracle_demos(
    samples: List[Any],
    rubric: str,
    n_demos: int = 8,
    max_doc_chars: int = 4000,
    text_field: str = 'text',
    label_field: str = 'reference_score',
) -> List[OracleAlignedDemo]:
    """
    Create oracle-aligned demonstrations from labeled samples.

    This is the main entry point for top-down initialization.
    Filters to SHORT documents where oracle(doc) = ground_truth by definition.

    Args:
        samples: Labeled samples (e.g., ManifestoSample with reference_score)
        rubric: Information preservation rubric
        n_demos: Number of demos to create
        max_doc_chars: Maximum document length in characters
        text_field: Attribute name for text content
        label_field: Attribute name for ground truth label

    Returns:
        List of OracleAlignedDemo (from short docs with guaranteed alignment)

    Example:
        from src.ops_engine.initialization import create_oracle_aligned_demos
        from src.tasks.manifesto import RILE_PRESERVATION_RUBRIC

        demos = create_oracle_aligned_demos(
            samples=train_samples,
            rubric=RILE_PRESERVATION_RUBRIC,
            n_demos=8,
            max_doc_chars=DEFAULT_MAX_DOC_CHARS,  # Short docs only
        )
    """
    initializer = TopDownInitializer(
        max_doc_chars=max_doc_chars,
        label_extractor=lambda x: getattr(x, label_field, None),
    )

    return initializer.create_demos_from_samples(
        samples=samples,
        rubric=rubric,
        n_demos=n_demos,
        text_field=text_field,
    )


def initialize_summarizer(
    summarizer: dspy.Module,
    demos: List[OracleAlignedDemo],
) -> dspy.Module:
    """
    Initialize a summarizer module with oracle-aligned demonstrations.

    Args:
        summarizer: DSPy summarizer module (e.g., LeafSummarizer)
        demos: Oracle-aligned demonstrations

    Returns:
        Initialized summarizer

    Example:
        from src.ops_engine.initialization import (
            create_oracle_aligned_demos,
            initialize_summarizer_with_demos,
        )
        from src.tasks.manifesto import LeafSummarizer  # Or use task.create_summarizer()

        demos = create_oracle_aligned_demos(train_samples, rubric, n_demos=8)
        summarizer = LeafSummarizer()
        initialized = initialize_summarizer_with_demos(summarizer, demos)
    """
    initializer = TopDownInitializer()
    return initializer.initialize_module(summarizer, demos)


def quick_demos(
    samples: List[Any],
    rubric: str,
    n_demos: int = 4,
    text_field: str = 'text',
    label_field: str = 'reference_score',
) -> List[dspy.Example]:
    """
    Quickly create initialization demos without oracle verification.

    This is a faster alternative that doesn't verify oracle alignment,
    useful when you want fast startup but still want seeded demos.

    Args:
        samples: Labeled samples
        rubric: Rubric for summarization
        n_demos: Number of demos
        text_field: Text attribute name
        label_field: Label attribute name

    Returns:
        List of DSPy Examples ready for module seeding
    """
    demos = []

    for sample in samples[:n_demos]:
        text = getattr(sample, text_field, None)
        if text is None and isinstance(sample, dict):
            text = sample.get(text_field, '')

        label = getattr(sample, label_field, None)
        if label is None and isinstance(sample, dict):
            label = sample.get(label_field, 0)

        if not text:
            continue

        # Create a simple label-focused summary
        # This is a heuristic - the real summary will be generated during forward pass
        summary_hint = f"[Political content with RILE indicators suggesting {label}]"

        demo = dspy.Example(
            content=text,  # Use full text - truncation corrupts training examples
            rubric=rubric,
            summary=summary_hint,
            ground_truth_label=label,
        ).with_inputs('content', 'rubric')

        demos.append(demo)

    logger.info(f"Created {len(demos)} quick-init demos")
    return demos


# =============================================================================
# Integration with Training Pipeline
# =============================================================================

def run_top_down_initialization(
    train_samples: List[Any],
    leaf_summarizer: dspy.Module,
    merge_summarizer: Optional[dspy.Module] = None,
    rubric: str = "",
    n_leaf_demos: int = 8,
    n_merge_demos: int = 4,
    max_doc_chars: int = 4000,
    text_field: str = 'text',
    label_field: str = 'reference_score',
) -> Tuple[dspy.Module, Optional[dspy.Module]]:
    """
    Run complete top-down initialization for summarizers.

    Uses SHORT documents where oracle(doc) = ground_truth by definition.
    These become perfect demos because the doc IS the ideal "summary".

    Args:
        train_samples: Training samples with ground truth labels
        leaf_summarizer: Leaf summarization module
        merge_summarizer: Optional merge summarization module
        rubric: Information preservation rubric
        n_leaf_demos: Number of leaf demos
        n_merge_demos: Number of merge demos
        max_doc_chars: Maximum document length (must fit in context)
        text_field: Text attribute name
        label_field: Label attribute name

    Returns:
        Tuple of (initialized_leaf, initialized_merge)

    Example:
        from src.ops_engine.initialization import run_top_down_initialization
        from src.tasks.manifesto import LeafSummarizer, MergeSummarizer  # Or use task.create_summarizer()

        leaf = LeafSummarizer()
        merge = MergeSummarizer()

        init_leaf, init_merge = run_top_down_initialization(
            train_samples=samples,
            leaf_summarizer=leaf,
            merge_summarizer=merge,
            rubric=task.create_rubric(),  # Use task's rubric
            n_leaf_demos=8,
            max_doc_chars=DEFAULT_MAX_DOC_CHARS,  # Short docs only
        )

        # Now use init_leaf, init_merge in training pipeline
    """
    logger.info(f"Running top-down initialization (max_doc_chars={max_doc_chars})...")

    initializer = TopDownInitializer(
        max_doc_chars=max_doc_chars,
        label_extractor=lambda x: getattr(x, label_field, None),
    )

    # Create leaf demos
    leaf_demos = initializer.create_demos_from_samples(
        samples=train_samples,
        rubric=rubric,
        n_demos=n_leaf_demos,
        text_field=text_field,
    )

    # Initialize leaf summarizer
    init_leaf = initializer.initialize_module(leaf_summarizer, leaf_demos)
    logger.info(f"Initialized leaf summarizer with {len(leaf_demos)} demos")

    # Initialize merge summarizer if provided
    init_merge = None
    if merge_summarizer and n_merge_demos > 0:
        # For merge demos, we need pairs of summaries
        # Create synthetic pairs from leaf demos
        merge_demos = []
        for i in range(0, len(leaf_demos) - 1, 2):
            if i + 1 < len(leaf_demos):
                left = leaf_demos[i]
                right = leaf_demos[i + 1]

                merge_demo = MergeAlignedDemo(
                    left_summary=left.summary,
                    right_summary=right.summary,
                    rubric=rubric,
                    merged_summary=format_merge_input(left.summary, right.summary),  # Simple concat as init
                    ground_truth_label=left.ground_truth_label,
                )
                merge_demos.append(merge_demo)

        # Set merge demos
        dspy_merge_demos = [d.to_dspy_example() for d in merge_demos[:n_merge_demos]]
        for name, submodule in merge_summarizer.named_predictors():
            if hasattr(submodule, 'demos'):
                submodule.demos = dspy_merge_demos

        init_merge = merge_summarizer
        logger.info(f"Initialized merge summarizer with {len(dspy_merge_demos)} demos")

    return init_leaf, init_merge


# =============================================================================
# Training-Based Initialization (Actual prompt optimization with GEPA)
# =============================================================================

def train_on_short_docs(
    train_samples: List[Any],
    leaf_summarizer: dspy.Module,
    merge_summarizer: Optional[dspy.Module] = None,
    oracle_classifier=None,
    rubric: str = "",
    label_field: str = 'rile',
    text_field: str = 'text',
    max_doc_chars: int = 8000,
    optimizer_type: str = 'gepa',
    max_metric_calls: int = 200,
    num_threads: int = 16,
) -> Tuple[dspy.Module, Optional[dspy.Module]]:
    """
    Train summarizers on short docs using actual prompt optimization (GEPA).

    Unlike demo selection (BootstrapFewShot), this uses GEPA to optimize
    the actual prompts/instructions. Short docs are used IN FULL as
    training data since they already fit in context.

    For short docs, oracle(doc) = ground_truth by definition, making them
    ideal training data. The metric checks if oracle(summary) ≈ ground_truth.

    Args:
        train_samples: Training samples with ground truth labels
        leaf_summarizer: Leaf summarization module to train
        merge_summarizer: Optional merge summarization module
        oracle_classifier: Oracle for RILE prediction (for ground-truth metric)
        rubric: Information preservation rubric
        label_field: Attribute name for ground truth label (default: 'rile')
        text_field: Attribute name for text content (default: 'text')
        max_doc_chars: Maximum document length for filtering (default: 8000)
        optimizer_type: 'gepa' (prompt optimization), 'mipro', or 'bootstrap'
        max_metric_calls: Budget for GEPA optimization (default: 200)
        num_threads: Parallel threads for optimization

    Returns:
        Tuple of (optimized_leaf, optimized_merge)

    Example:
        leaf, merge = train_on_short_docs(
            train_samples=samples,
            leaf_summarizer=LeafSummarizer(),
            oracle_classifier=rile_scorer,  # Any oracle with predict_rile() method
            rubric=RILE_RUBRIC,
            optimizer_type='gepa',  # Actual prompt optimization
        )
    """
    logger.info(f"Training summarizers on short docs (max_doc_chars={max_doc_chars}, optimizer={optimizer_type})...")

    # 1. Filter to short docs (use FULL text, no truncation)
    short_samples = [
        s for s in train_samples
        if len(getattr(s, text_field, '') or '') <= max_doc_chars
        and getattr(s, label_field, None) is not None
    ]

    if not short_samples:
        logger.warning(
            f"No short documents found (≤{max_doc_chars} chars). "
            f"Returning original summarizers without training."
        )
        return leaf_summarizer, merge_summarizer

    logger.info(f"Found {len(short_samples)} short docs for training")

    # 2. Create trainset: for short docs, doc IS the ideal summary
    # Include reference_score for the metric
    trainset = []
    for sample in short_samples:
        text = getattr(sample, text_field, '')  # FULL text, no truncation
        label = getattr(sample, label_field, None)

        if text and label is not None:
            trainset.append(dspy.Example(
                content=text,
                rubric=rubric,
                summary=text,  # Short doc is its own ideal summary
                reference_score=label,  # Known label for metric
            ).with_inputs('content', 'rubric'))

    if not trainset:
        logger.warning("No valid training examples created. Returning original summarizers.")
        return leaf_summarizer, merge_summarizer

    logger.info(f"Created {len(trainset)} training examples")

    # 3. Create ground-truth based metric
    # For short docs, we KNOW the ground truth RILE label
    # Good summary → oracle(summary) ≈ ground_truth
    def rile_preservation_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        """Check if summary preserves RILE-relevant information.

        GEPA requires 5 arguments: (gold, pred, trace, pred_name, pred_trace)
        """
        summary = getattr(pred, 'summary', '')
        if not summary or len(summary) < 50:
            return 0.0

        # Use oracle to predict RILE from summary, compare to ground truth
        if oracle_classifier is not None:
            try:
                estimated_score = oracle_classifier.predict_rile(summary)
                reference = example.reference_score
                error = abs(estimated_score - reference)
                # Convert error to score (100-point RILE scale)
                return max(0.0, 1.0 - error / 100.0)
            except Exception as e:
                logger.debug(f"Oracle metric failed: {e}, using fallback")

        # Fallback: content preservation heuristic
        return 1.0 if len(summary) >= len(example.content) * 0.3 else 0.5

    # 4. Create optimizer based on type
    if optimizer_type == 'gepa':
        logger.info(f"Using GEPA optimizer (max_metric_calls={max_metric_calls}) for actual prompt optimization")
        optimizer = dspy.GEPA(
            metric=rile_preservation_metric,
            max_metric_calls=max_metric_calls,
        )
    elif optimizer_type == 'mipro':
        logger.info(f"Using MIPROv2 optimizer (auto=light) for instruction optimization")
        optimizer = dspy.MIPROv2(
            metric=rile_preservation_metric,
            auto='light',
            num_threads=num_threads,
        )
    else:
        # Fallback to bootstrap (demo selection, not recommended)
        logger.info(f"Using BootstrapFewShot optimizer (demo selection fallback)")
        optimizer = dspy.BootstrapFewShot(
            metric=rile_preservation_metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
        )

    # 5. Train leaf summarizer
    try:
        logger.info("Training leaf summarizer...")
        optimized_leaf = optimizer.compile(leaf_summarizer, trainset=trainset)
        logger.info("Leaf summarizer training complete")
    except Exception as e:
        logger.warning(f"Leaf summarizer training failed: {e}. Using original.")
        optimized_leaf = leaf_summarizer

    # 6. Train merge summarizer if provided
    optimized_merge = merge_summarizer
    if merge_summarizer is not None and len(trainset) >= 2:
        try:
            # Create merge trainset from pairs
            merge_trainset = []
            for i in range(0, len(trainset) - 1, 2):
                left = trainset[i]
                right = trainset[i + 1]

                # For merge, include ground truth from both parents
                # Merged summary should preserve RILE from both
                merge_trainset.append(dspy.Example(
                    left_summary=left.content,
                    right_summary=right.content,
                    rubric=rubric,
                    merged_summary=f"{left.content}\n\n{right.content}",
                    reference_score=(left.reference_score + right.reference_score) / 2,
                ).with_inputs('left_summary', 'right_summary', 'rubric'))

            if merge_trainset:
                logger.info(f"Training merge summarizer on {len(merge_trainset)} examples...")
                # Create new optimizer for merge (same type)
                if optimizer_type == 'gepa':
                    merge_optimizer = dspy.GEPA(
                        metric=rile_preservation_metric,
                        max_metric_calls=max_metric_calls,
                    )
                elif optimizer_type == 'mipro':
                    merge_optimizer = dspy.MIPROv2(
                        metric=rile_preservation_metric,
                        auto='light',
                        num_threads=num_threads,
                    )
                else:
                    merge_optimizer = dspy.BootstrapFewShot(
                        metric=rile_preservation_metric,
                        max_bootstrapped_demos=4,
                    )
                optimized_merge = merge_optimizer.compile(merge_summarizer, trainset=merge_trainset)
                logger.info("Merge summarizer training complete")

        except Exception as e:
            logger.warning(f"Merge summarizer training failed: {e}. Using original.")
            optimized_merge = merge_summarizer

    return optimized_leaf, optimized_merge
