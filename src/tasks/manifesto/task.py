"""
Manifesto/RILE task plugin.

This module implements the TaskPlugin interface for political manifesto
analysis using the RILE (Right-Left) scoring system from the Manifesto Project.

The manifesto task uses:
- Continuous score prediction (not classification)
- RILE-specific rubrics and task contexts
- Score-based metrics using ScaleDefinition
"""

import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import dspy

from src.ops_engine.training_framework.tasks.base import AbstractTask, OutputType, TaskConfig, ScaleDefinition
from src.tasks.prompting import PromptBuilders
from src.ops_engine.training_framework.tasks.registry import register_task

if TYPE_CHECKING:
    from src.ops_engine.training_framework.core import TrainingDataSource
    from src.ops_engine.training_framework.config import OracleIRRConfig
    from src.ops_engine.training_framework.inference import Retriever

logger = logging.getLogger(__name__)


# RILE scale definition - specific to manifesto task
RILE_SCALE = ScaleDefinition(
    name="rile",
    min_value=-100.0,
    max_value=100.0,
    description="Right-Left ideological scale. -100 = far left, +100 = far right",
    higher_is_better=True,  # Higher scores don't mean "better", but we use this for optimization
    neutral_value=0.0,
)


@register_task(["manifesto_rile", "rile_scoring"])
class ManifestoTask(AbstractTask):
    """
    Manifesto/RILE task implementation.

    This task handles political manifesto analysis using the RILE
    (Right-Left) ideological scoring system.

    RILE scores range from -100 (far left) to +100 (far right).
    Uses continuous score prediction rather than classification.
    """

    def __init__(
        self,
        error_threshold_high: float = 20.0,
        error_threshold_low: float = 10.0,
    ):
        """
        Initialize manifesto task.

        Args:
            error_threshold_high: Error above which is labeled as violation
            error_threshold_low: Error below which is labeled as good
        """
        super().__init__()
        self.error_threshold_high = error_threshold_high
        self.error_threshold_low = error_threshold_low

        # Set up task configuration using the new abstractions
        self._config = TaskConfig(
            name="manifesto_rile",
            output_type=OutputType.CONTINUOUS_SCORE,
            scale=RILE_SCALE,
            output_field_name="rile_score",
            rubric_template="",  # Will be filled by create_rubric
            task_context_template="",  # Will be filled by get_task_context
        )

    @property
    def name(self) -> str:
        return "manifesto_rile"

    # =========================================================================
    # Data Field Configuration (implements AbstractTask interface)
    # =========================================================================

    @property
    def id_field(self) -> str:
        """Field name for document ID in manifesto samples."""
        return "manifesto_id"

    @property
    def label_field(self) -> str:
        """Field name for ground truth RILE score."""
        return "rile"

    @property
    def text_field(self) -> str:
        """Field name for document text."""
        return "text"

    # =========================================================================
    # Data Loading (implements AbstractTask interface)
    # =========================================================================

    def get_data_loader(self):
        """Get ManifestoDataLoader instance."""
        from .data_loader import ManifestoDataLoader
        return ManifestoDataLoader()

    def get_samples(
        self,
        splits: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get manifesto samples from the data loader.

        Args:
            splits: List of splits to load. Defaults to ["train", "val"].
                   Valid splits: "train", "val", "test"
            limit: Maximum number of samples to return.

        Returns:
            List of sample dictionaries with manifesto_id, text, rile fields.
        """
        loader = self.get_data_loader()
        train, val, test = loader.get_temporal_split()

        samples = []
        splits = splits or ["train", "val"]

        if "train" in splits:
            samples.extend(train)
        if "val" in splits:
            samples.extend(val)
        if "test" in splits:
            samples.extend(test)

        if limit:
            samples = samples[:limit]

        # Convert to dictionaries if needed
        result = []
        for s in samples:
            if hasattr(s, '__dict__'):
                result.append({
                    self.id_field: getattr(s, self.id_field, None),
                    self.text_field: getattr(s, self.text_field, ''),
                    self.label_field: getattr(s, self.label_field, None),
                })
            else:
                result.append(s)
        return result

    def create_training_source(
        self,
        results: List[Any],
        **kwargs,
    ) -> 'TrainingDataSource':
        """
        Create a training data source from manifesto processing results.

        Args:
            results: List of ManifestoResult objects
            **kwargs: Additional configuration (error_threshold_high, error_threshold_low)

        Returns:
            UnifiedTrainingSource instance configured for RILE scale
        """
        from src.ops_engine.training_framework.tasks.base import UnifiedTrainingSource

        # Convert absolute thresholds to normalized (0-1) scale
        error_high = kwargs.get('error_threshold_high', self.error_threshold_high)
        error_low = kwargs.get('error_threshold_low', self.error_threshold_low)

        # Normalize thresholds to scale range (200 points for RILE)
        error_high_normalized = error_high / RILE_SCALE.range
        error_low_normalized = error_low / RILE_SCALE.range

        source = UnifiedTrainingSource(
            error_threshold_high=error_high_normalized,
            error_threshold_low=error_low_normalized,
            rubric=self.create_rubric(),
            source_name="manifesto_rile",
            scale=RILE_SCALE,
        )
        source.add_results(results)
        return source

    def create_metric(
        self,
        with_feedback: bool = True,
    ) -> Callable:
        """
        Create a DSPy-compatible metric for RILE score prediction.

        Args:
            with_feedback: Whether to include feedback for GEPA reflection

        Returns:
            Metric function
        """
        from src.ops_engine.scoring import oracle_as_metric_with_feedback, oracle_as_metric

        # Return the appropriate metric based on feedback preference
        if with_feedback:
            return oracle_as_metric_with_feedback
        return oracle_as_metric

    def create_prompt_builders(self) -> PromptBuilders:
        """Create prompt builders for RILE scoring."""
        return PromptBuilders(
            summarize=_build_rile_summarize_prompt,
            merge=_build_rile_merge_prompt,
            score=_build_rile_score_prompt,
            audit=_build_rile_audit_prompt,
        )

    def parse_score(self, response: str) -> Optional[float]:
        """Parse a RILE score (-100 to +100) from text."""
        import re

        if not response:
            return None

        match = re.search(r'RILE_SCORE:\s*(-?\d+\.?\d*)', response, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        numbers = re.findall(r'-?\d+\.?\d*', response)
        for num_str in numbers:
            try:
                num = float(num_str)
                if -100 <= num <= 100:
                    return num
            except ValueError:
                continue

        return None

    def create_rubric(self, **kwargs) -> str:
        """
        Create a RILE preservation rubric.

        Args:
            **kwargs: Rubric customization options
                - detailed: Include detailed examples (default: True)
                - include_indicators: Include left/right indicator lists

        Returns:
            RILE rubric string
        """
        # Import the existing rubric or create a parameterized version
        try:
            from .rubrics import RILE_PRESERVATION_RUBRIC
            return RILE_PRESERVATION_RUBRIC
        except ImportError:
            # Fallback to a basic rubric
            return self._create_basic_rubric(**kwargs)

    def _create_basic_rubric(self, **kwargs) -> str:
        """Create a basic RILE rubric if the detailed one isn't available."""
        return """
RILE (Right-Left) Position Preservation Rubric

This rubric evaluates whether a summary preserves the political positioning
signals from the original text.

Key Dimensions:
1. Economic Policy: market economy vs. state intervention
2. Social Policy: traditional values vs. progressive values
3. Foreign Policy: nationalism vs. internationalism

Evaluation Criteria:
- Does the summary preserve indicators of left-leaning positions?
- Does the summary preserve indicators of right-leaning positions?
- Is the overall political positioning maintained?

Scoring:
- The summary should enable accurate RILE scoring
- Key policy positions should be preserved
- Political rhetoric and framing should be maintained
""".strip()

    def get_task_context(self) -> str:
        """
        Get the task context for RILE analysis.

        Returns:
            Task context string
        """
        try:
            from .rubrics import RILE_TASK_CONTEXT
            return RILE_TASK_CONTEXT
        except ImportError:
            return """
Analyze political manifesto text to determine RILE (Right-Left) positioning.
RILE scores range from -100 (far left) to +100 (far right).
Focus on economic policy, social values, and foreign policy positions.
""".strip()

    def create_predictor(
        self,
        retriever: Optional['Retriever'] = None,
        config: Optional['OracleIRRConfig'] = None,
    ) -> dspy.Module:
        """
        Create a RILE score predictor module.

        Args:
            retriever: Optional retriever for retrieval-augmented prediction
            config: Optional configuration

        Returns:
            DSPy Module for RILE score prediction (RILEScorer)
        """
        from .dspy_signatures import RILEScorer
        return RILEScorer()

    def create_oracle_scorer(self) -> Callable[[str], float]:
        """
        Create an oracle scorer function for tournament of tournaments.

        Returns a function that takes a text and returns a RILE score.
        This is used to determine which summaries lead to better downstream
        task performance during judge optimization.

        The oracle scorer uses RILEScorer to predict RILE scores for summaries.
        During training, we compare oracle(summary) error vs ground truth to
        determine which summary is objectively better.

        Returns:
            Function(text) -> RILE score (-100 to +100)
        """
        from .dspy_signatures import RILEScorer

        scorer = RILEScorer()
        task_context = self.get_task_context()

        def oracle_predict(text: str) -> float:
            """Predict RILE score for text."""
            try:
                result = scorer(text=text, task_context=task_context)
                return float(result.get('rile_score', 0.0))
            except Exception as e:
                logger.warning(f"Oracle prediction failed: {e}")
                return 0.0  # Neutral score on error

        return oracle_predict

    def create_summarizer(self) -> dspy.Module:
        """Create a summarizer tuned for RILE preservation."""
        from .summarizer import GenericSummarizer
        return GenericSummarizer(use_cot=True)

    def create_merge_summarizer(self) -> dspy.Module:
        """Create a merge summarizer tuned for RILE preservation."""
        return self.create_summarizer()

    def get_info(self) -> Dict[str, Any]:
        """Get information about this task."""
        base_info = super().get_info()
        base_info.update({
            'error_threshold_high': self.error_threshold_high,
            'error_threshold_low': self.error_threshold_low,
            'score_type': 'continuous',
            'score_range': '[-100, +100]',
        })
        return base_info


def _build_rile_summarize_prompt(text: str, rubric: str):
    """Build RILE-aware summarization prompt."""
    return [
        {
            "role": "system",
            "content": "Political text summarizer. Preserve left-right positioning info.",
        },
        {
            "role": "user",
            "content": (
                f"Summarize per rubric:\n{rubric}\n\n"
                f"Text:\n{text}\n\n"
                "Summary:"
            ),
        },
    ]


def _build_rile_merge_prompt(left: str, right: str, rubric: str):
    """Build RILE-aware merge prompt."""
    return [
        {
            "role": "system",
            "content": "Political text summarizer. Preserve all politically relevant info.",
        },
        {
            "role": "user",
            "content": (
                f"Merge per rubric:\n{rubric}\n\n"
                f"Summary 1:\n{left}\n\n"
                f"Summary 2:\n{right}\n\n"
                "Merged:"
            ),
        },
    ]


def _build_rile_score_prompt(text: str, task_context: str):
    """Build RILE scoring prompt."""
    return [
        {
            "role": "system",
            "content": "Political scientist. Score manifestos on RILE scale (-100 to +100).",
        },
        {
            "role": "user",
            "content": (
                f"{task_context}\n\n"
                f"Text:\n{text}\n\n"
                "Score (-100 to +100):\nRILE_SCORE: <number>\nReasoning:"
            ),
        },
    ]


def _build_rile_audit_prompt(original: str, summary: str, rubric: str):
    """Build audit/oracle prompt for RILE preservation."""
    return [
        {
            "role": "system",
            "content": (
                "You are auditing whether a summary preserves politically relevant "
                "information from the original text."
            ),
        },
        {
            "role": "user",
            "content": (
                "Does the summary preserve the political position information "
                "from the original?\n\n"
                f"Criteria: {rubric}\n\n"
                f"ORIGINAL:\n{original}\n\n"  # Use full text - truncation corrupts evaluation
                f"SUMMARY:\n{summary}\n\n"
                "Answer PASS if information is preserved, FAIL if not. "
                "Format: VERDICT: PASS/FAIL\nREASON: <explanation>"
            ),
        },
    ]

