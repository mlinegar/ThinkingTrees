"""
Manifesto/RILE domain plugin.

This module implements the DomainPlugin interface for political manifesto
analysis using the RILE (Right-Left) scoring system from the Manifesto Project.

The manifesto domain uses:
- Ordinal label space with configurable bin sizes
- RILE-specific rubrics and task contexts
- Distance-weighted metrics for ordinal scoring
"""

import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .base import AbstractDomain
from .registry import register_domain
from ..core import OrdinalLabelSpace, LabelSpace

if TYPE_CHECKING:
    import dspy
    from ..core import TrainingDataSource
    from ..config import OracleIRRConfig
    from ..inference import Retriever, RILEClassifier

logger = logging.getLogger(__name__)


@register_domain("manifesto_rile")
class ManifestoDomain(AbstractDomain):
    """
    Manifesto/RILE domain implementation.

    This domain handles political manifesto analysis using the RILE
    (Right-Left) ideological scoring system.

    RILE scores range from -100 (far left) to +100 (far right), with
    the label space discretized into bins for classification.
    """

    def __init__(
        self,
        bin_size: float = 10.0,
        error_threshold_high: float = 20.0,
        error_threshold_low: float = 10.0,
    ):
        """
        Initialize manifesto domain.

        Args:
            bin_size: Size of RILE bins for discretization (default: 10.0)
            error_threshold_high: Error above which is labeled as violation
            error_threshold_low: Error below which is labeled as good
        """
        super().__init__()
        self.bin_size = bin_size
        self.error_threshold_high = error_threshold_high
        self.error_threshold_low = error_threshold_low
        self._label_space = OrdinalLabelSpace.for_rile(bin_size)

    @property
    def name(self) -> str:
        return "manifesto_rile"

    @property
    def label_space(self) -> LabelSpace:
        return self._label_space

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
            ManifestoTrainingSource instance
        """
        # Import here to avoid circular imports
        from src.manifesto.training_integration import ManifestoTrainingSource

        error_high = kwargs.get('error_threshold_high', self.error_threshold_high)
        error_low = kwargs.get('error_threshold_low', self.error_threshold_low)

        source = ManifestoTrainingSource(
            label_space=self._label_space,
            error_threshold_high=error_high,
            error_threshold_low=error_low,
        )
        source.add_results(results)
        return source

    def create_metric(
        self,
        weighted: bool = True,
        with_feedback: bool = True,
    ) -> Callable:
        """
        Create a DSPy-compatible metric for RILE classification.

        Args:
            weighted: Whether to use distance-weighted scoring
            with_feedback: Whether to include feedback for GEPA reflection

        Returns:
            Metric function
        """
        from ..metrics import create_classification_metric

        return create_classification_metric(
            label_space=self._label_space,
            weighted=weighted,
            with_feedback=with_feedback,
        )

    def create_classifier(
        self,
        retriever: Optional['Retriever'] = None,
        config: Optional['OracleIRRConfig'] = None,
    ) -> 'dspy.Module':
        """
        Create a RILE classifier module.

        Args:
            retriever: Optional retriever for retrieval-augmented classification
            config: Optional Oracle IRR configuration

        Returns:
            RILEClassifier module
        """
        from ..inference import RILEClassifier
        from ..config import OracleIRRConfig

        config = config or OracleIRRConfig()

        return RILEClassifier(
            bin_size=self.bin_size,
            retriever=retriever,
            config=config,
        )

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
            from src.manifesto.rubrics import RILE_PRESERVATION_RUBRIC
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
            from src.manifesto.rubrics import RILE_TASK_CONTEXT
            return RILE_TASK_CONTEXT
        except ImportError:
            return """
Analyze political manifesto text to determine RILE (Right-Left) positioning.
RILE scores range from -100 (far left) to +100 (far right).
Focus on economic policy, social values, and foreign policy positions.
""".strip()

    def get_info(self) -> Dict[str, Any]:
        """Get information about this domain."""
        base_info = super().get_info()
        base_info.update({
            'bin_size': self.bin_size,
            'error_threshold_high': self.error_threshold_high,
            'error_threshold_low': self.error_threshold_low,
            'num_labels': len(self._label_space.labels) if hasattr(self._label_space, 'labels') else 'unknown',
        })
        return base_info
