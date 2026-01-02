"""
Manifesto RILE Pipeline Components.

DSPy modules and pipelines for processing political manifestos through
OPS trees with RILE scoring.

These components are manifesto-specific and use RILE preservation rubrics.
"""

import logging
from typing import Optional, TYPE_CHECKING

import dspy

from src.tree.builder import TreeBuilder, BuildConfig
from .rubrics import RILE_PRESERVATION_RUBRIC, RILE_TASK_CONTEXT
from .constants import RILE_MIN, RILE_MAX

if TYPE_CHECKING:
    from src.core.strategy import DSPyStrategy, TournamentStrategy, TournamentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# DSPy Signatures
# =============================================================================

class RILESummarize(dspy.Signature):
    """Summarize political text preserving information relevant to left-right positioning."""
    rubric: str = dspy.InputField(desc="What information to preserve for RILE scoring")
    text: str = dspy.InputField(desc="Political text chunk to summarize")

    summary: str = dspy.OutputField(desc="Concise summary preserving left/right position indicators")


class RILEMerge(dspy.Signature):
    """Merge two summaries while preserving political position information."""
    rubric: str = dspy.InputField(desc="What information to preserve for RILE scoring")
    summary1: str = dspy.InputField(desc="First summary to merge")
    summary2: str = dspy.InputField(desc="Second summary to merge")

    merged_summary: str = dspy.OutputField(desc="Combined summary preserving all position indicators from both inputs")


class RILEScoreSignature(dspy.Signature):
    """Score political text on left-right scale."""
    task_context: str = dspy.InputField(desc="The RILE scoring task and scale explanation")
    summary: str = dspy.InputField(desc="Summarized political manifesto to score")

    reasoning: str = dspy.OutputField(desc="Analysis identifying left vs right indicators and their balance")
    rile_score: float = dspy.OutputField(desc="RILE score from -100 (far left) to +100 (far right)")


# =============================================================================
# Helper Functions
# =============================================================================

def is_placeholder(text: str) -> bool:
    """Check if text is a template placeholder instead of real content."""
    if not text or len(text) < 50:
        placeholders = ['[', ']', 'summary', 'merged', 'content', 'text here']
        text_lower = text.lower()
        return any(p in text_lower for p in placeholders)
    return False


# =============================================================================
# DSPy Modules
# =============================================================================

class ManifestoSummarizer(dspy.Module):
    """DSPy module for summarizing chunks - optimizable by DSPy."""

    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought(RILESummarize)

    def forward(self, text: str, rubric: str = RILE_PRESERVATION_RUBRIC) -> str:
        result = self.summarize(rubric=rubric, text=text)
        summary = result.summary

        if is_placeholder(summary):
            logger.warning(f"Got placeholder summary: {summary[:50]}... Retrying...")
            result = self.summarize(rubric=rubric, text=text)
            summary = result.summary
            if is_placeholder(summary):
                # Raise error instead of truncated fallback - truncation corrupts data
                raise ValueError(f"Failed to generate valid summary after retry. Got placeholder: {summary}")

        return summary


class ManifestoMerger(dspy.Module):
    """DSPy module for merging summaries - optimizable by DSPy."""

    def __init__(self):
        super().__init__()
        self.merge = dspy.ChainOfThought(RILEMerge)

    def forward(self, summary1: str, summary2: str, rubric: str = RILE_PRESERVATION_RUBRIC) -> str:
        result = self.merge(rubric=rubric, summary1=summary1, summary2=summary2)
        merged = result.merged_summary

        if is_placeholder(merged):
            logger.warning(f"Got placeholder merge: {merged[:50]}... Retrying...")
            result = self.merge(rubric=rubric, summary1=summary1, summary2=summary2)
            merged = result.merged_summary
            if is_placeholder(merged):
                merged = f"{summary1}\n\n{summary2}"

        return merged


class ManifestoScorer(dspy.Module):
    """DSPy module for scoring - optimizable by DSPy."""

    def __init__(self):
        super().__init__()
        self.score = dspy.ChainOfThought(RILEScoreSignature)

    def forward(self, summary: str, task_context: str = RILE_TASK_CONTEXT) -> dict:
        result = self.score(task_context=task_context, summary=summary)
        try:
            raw_score = float(result.rile_score)
        except (ValueError, TypeError):
            raw_score = 0.0
        raw_score = max(RILE_MIN, min(RILE_MAX, raw_score))
        normalized = (raw_score - RILE_MIN) / (RILE_MAX - RILE_MIN)
        normalized = max(0.0, min(1.0, normalized))
        return {
            'rile_score': normalized,
            'reasoning': result.reasoning
        }


# =============================================================================
# Strategy-Compatible Wrappers
# =============================================================================

class StrategyCompatibleSummarizer(dspy.Module):
    """
    DSPy summarizer compatible with DSPyStrategy parameter names.

    Wraps ManifestoSummarizer and translates parameter names:
    - content -> text
    - rubric -> rubric (unchanged)
    """

    def __init__(self):
        super().__init__()
        self._inner = ManifestoSummarizer()

    def forward(self, content: str, rubric: str = RILE_PRESERVATION_RUBRIC) -> str:
        """Forward with DSPyStrategy-compatible parameter names."""
        return self._inner(text=content, rubric=rubric)


class StrategyCompatibleMerger(dspy.Module):
    """
    DSPy merger compatible with DSPyStrategy parameter names.

    Wraps ManifestoMerger and translates parameter names:
    - left_summary -> summary1
    - right_summary -> summary2
    - rubric -> rubric (unchanged)
    """

    def __init__(self):
        super().__init__()
        self._inner = ManifestoMerger()

    def forward(self, left_summary: str, right_summary: str, rubric: str = RILE_PRESERVATION_RUBRIC) -> str:
        """Forward with DSPyStrategy-compatible parameter names."""
        return self._inner(summary1=left_summary, summary2=right_summary, rubric=rubric)


# =============================================================================
# Full Pipelines
# =============================================================================

class ManifestoPipeline(dspy.Module):
    """
    Full DSPy pipeline: chunk -> summarize -> merge -> score.
    The entire pipeline is optimizable by DSPy.
    Uses parallel processing for chunk summarization and merging.
    """

    def __init__(self, chunk_size: int = 2000):
        super().__init__()
        self.chunk_size = chunk_size
        self.summarizer = ManifestoSummarizer()
        self.merger = ManifestoMerger()
        self.scorer = ManifestoScorer()

    def forward(self, text: str, rubric: str = RILE_PRESERVATION_RUBRIC,
                task_context: str = RILE_TASK_CONTEXT) -> dspy.Prediction:
        """Process a full manifesto through the pipeline with parallel execution."""
        from src.preprocessing.chunker import chunk_for_ops
        from concurrent.futures import ThreadPoolExecutor, as_completed

        chunks = chunk_for_ops(text, max_chars=self.chunk_size, strategy="sentence")

        if not chunks:
            return dspy.Prediction(rile_score=0.5, reasoning="No text to process", final_summary="")

        def summarize_chunk(chunk_text):
            return self.summarizer(text=chunk_text, rubric=rubric)

        summaries = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            future_to_idx = {
                executor.submit(summarize_chunk, chunk.text): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    summaries[idx] = future.result()
                except Exception as e:
                    # Re-raise instead of truncated fallback - truncation corrupts data
                    logger.error(f"Chunk {idx} summarization failed: {e}")
                    raise

        while len(summaries) > 1:
            pairs = []
            odd_summary = None
            for i in range(0, len(summaries), 2):
                if i + 1 < len(summaries):
                    pairs.append((summaries[i], summaries[i+1]))
                else:
                    odd_summary = summaries[i]

            next_level = [None] * len(pairs)
            if pairs:
                def merge_pair(s1, s2):
                    return self.merger(summary1=s1, summary2=s2, rubric=rubric)

                with ThreadPoolExecutor(max_workers=len(pairs)) as executor:
                    future_to_idx = {
                        executor.submit(merge_pair, s1, s2): i
                        for i, (s1, s2) in enumerate(pairs)
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            next_level[idx] = future.result()
                        except Exception:
                            s1, s2 = pairs[idx]
                            next_level[idx] = f"{s1}\n\n{s2}"

            summaries = [s for s in next_level if s is not None]
            if odd_summary is not None:
                summaries.append(odd_summary)

        final_summary = summaries[0] if summaries else ""
        score_result = self.scorer(summary=final_summary, task_context=task_context)

        return dspy.Prediction(
            rile_score=score_result['rile_score'],
            reasoning=score_result['reasoning'],
            final_summary=final_summary
        )


class ManifestoPipelineWithStrategy(dspy.Module):
    """
    DSPy pipeline using the strategy pattern with TreeBuilder.

    This pipeline:
    1. Uses DSPyStrategy to wrap DSPy modules
    2. Uses TreeBuilder for tree construction
    3. Can optionally use TournamentStrategy for preference collection

    The modules remain optimizable by DSPy while benefiting from:
    - Unified tree-building logic
    - Optional tournament selection and preference collection
    - Cleaner separation of concerns

    Usage:
        # Basic usage
        pipeline = ManifestoPipelineWithStrategy()
        result = pipeline(text="...")

        # With tournament selection (for learning)
        pipeline = ManifestoPipelineWithStrategy(judge=genrm_judge)
        result = pipeline(text="...")
        preferences = pipeline.get_preferences()
    """

    def __init__(
        self,
        chunk_size: int = 2000,
        judge=None,
        tournament_k: int = 4,
        tournament_temperature: float = 0.9,
        leaf_module: Optional[dspy.Module] = None,
        merge_module: Optional[dspy.Module] = None,
        scorer: Optional[dspy.Module] = None,
    ):
        """
        Initialize the strategy-based pipeline.

        Args:
            chunk_size: Maximum chunk size for text splitting
            judge: Optional GenRMJudge or GenRMComparisonModule for tournament selection
            tournament_k: Number of candidates for tournament (default 4)
            tournament_temperature: Temperature for candidate generation (default 0.9)
            leaf_module: Optional leaf summarizer module (content, rubric -> summary)
            merge_module: Optional merge summarizer module (left_summary, right_summary, rubric -> summary)
            scorer: Optional scorer module (summary, task_context -> dict/prediction)
        """
        super().__init__()
        self.chunk_size = chunk_size

        self.leaf_module = leaf_module or StrategyCompatibleSummarizer()
        self.merge_module = merge_module or StrategyCompatibleMerger()
        self.scorer = scorer or ManifestoScorer()

        self._judge = judge
        self._tournament_k = tournament_k
        self._tournament_temperature = tournament_temperature

        self._last_strategy = None
        self._last_builder = None

    def _create_strategy(self):
        """Create the strategy stack (called per forward pass)."""
        # Lazy import to avoid circular dependency
        from src.core.strategy import DSPyStrategy, TournamentStrategy, TournamentConfig

        base_strategy = DSPyStrategy(
            leaf_module=self.leaf_module,
            merge_module=self.merge_module,
        )

        if self._judge is not None:
            config = TournamentConfig(
                k=self._tournament_k,
                temperature=self._tournament_temperature,
            )
            return TournamentStrategy(base=base_strategy, judge=self._judge, config=config)
        else:
            return base_strategy

    def forward(
        self,
        text: str,
        rubric: str = RILE_PRESERVATION_RUBRIC,
        task_context: str = RILE_TASK_CONTEXT,
    ) -> dspy.Prediction:
        """
        Process a manifesto through the strategy-based pipeline.

        Args:
            text: Manifesto text to process
            rubric: Information preservation criteria
            task_context: Task context for RILE scoring

        Returns:
            dspy.Prediction with rile_score, reasoning, final_summary
        """
        if not text or len(text.strip()) == 0:
            return dspy.Prediction(
                rile_score=0.5,
                reasoning="No text to process",
                final_summary=""
            )

        strategy = self._create_strategy()
        self._last_strategy = strategy

        config = BuildConfig(max_chunk_chars=self.chunk_size)
        builder = TreeBuilder(strategy=strategy, config=config)
        self._last_builder = builder

        try:
            result = builder.build_sync(text, rubric)
            final_summary = result.tree.root.summary
        except Exception as e:
            # Re-raise instead of truncated fallback - truncation corrupts data
            logger.error(f"Tree building failed: {e}")
            raise

        try:
            score_result = self.scorer(summary=final_summary, task_context=task_context)
        except TypeError:
            score_result = self.scorer(text=final_summary, task_context=task_context)

        score_value = None
        reasoning = ""
        if isinstance(score_result, dict):
            if "rile_score" in score_result:
                score_value = score_result.get("rile_score")
            elif "score" in score_result:
                score_value = score_result.get("score")
            reasoning = score_result.get("reasoning", "") or ""
        else:
            if hasattr(score_result, "rile_score"):
                score_value = getattr(score_result, "rile_score")
            elif hasattr(score_result, "score"):
                score_value = getattr(score_result, "score")
            reasoning = getattr(score_result, "reasoning", "") or ""

        if score_value is None:
            score_value = 0.5

        return dspy.Prediction(
            rile_score=float(score_value),
            reasoning=reasoning,
            final_summary=final_summary
        )

    def get_preferences(self):
        """Get collected preferences from tournament selection."""
        if self._last_strategy is not None and hasattr(self._last_strategy, 'get_preferences'):
            return self._last_strategy.get_preferences()
        return []

    def reset_preferences(self):
        """Reset collected preferences between documents."""
        if self._last_strategy is not None and hasattr(self._last_strategy, 'reset_preferences'):
            self._last_strategy.reset_preferences()


# =============================================================================
# Training Helpers
# =============================================================================

def create_training_examples(samples: list) -> list:
    """Create DSPy training examples from samples with ground truth."""
    examples = []
    for sample in samples:
        normalized_score = (sample.rile - RILE_MIN) / (RILE_MAX - RILE_MIN)
        normalized_score = max(0.0, min(1.0, normalized_score))
        example = dspy.Example(
            text=sample.text,
            rubric=RILE_PRESERVATION_RUBRIC,
            task_context=RILE_TASK_CONTEXT,
            rile_score=normalized_score,
        ).with_inputs('text', 'rubric', 'task_context')
        examples.append(example)
    return examples


def rile_metric(example, prediction, trace=None) -> float:
    """
    DSPy metric: how close is prediction to ground truth RILE?
    Returns 1.0 for perfect, 0.0 for >=1.0 normalized difference.
    """
    try:
        pred_score = float(prediction.rile_score)
        true_score = float(example.rile_score)
        error = abs(pred_score - true_score)
        return max(0.0, 1.0 - error)
    except (ValueError, TypeError, AttributeError):
        return 0.0
