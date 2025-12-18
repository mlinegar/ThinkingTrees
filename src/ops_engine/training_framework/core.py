"""
Core types and protocols for the Oracle Approximation Training Framework.

This module defines the fundamental data structures and interfaces used
throughout the training framework.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, List, Optional, Dict, Any, runtime_checkable

import dspy


class ViolationType(Enum):
    """Categories of OPS law violations (ontology)."""
    SUFFICIENCY = "sufficiency"        # C1: leaf doesn't preserve rubric info
    MERGE_CONSISTENCY = "merge"        # C3B: internal merge loses info
    IDEMPOTENCE = "idempotence"        # C2: re-summarization changes oracle
    SUBSTITUTION = "substitution"      # C3A: boundary inconsistency
    NONE = "none"                      # No violation (false positive)

    @classmethod
    def from_check_type(cls, check_type: str) -> 'ViolationType':
        """Map audit check type to violation type."""
        mapping = {
            'sufficiency': cls.SUFFICIENCY,
            'merge_consistency': cls.MERGE_CONSISTENCY,
            'merge': cls.MERGE_CONSISTENCY,
            'idempotence': cls.IDEMPOTENCE,
            'substitution': cls.SUBSTITUTION,
        }
        return mapping.get(check_type.lower(), cls.SUFFICIENCY)


class TrainingExampleLabel(Enum):
    """Label for training examples."""
    POSITIVE = "positive"   # True violation - confirmed needs fixing
    NEGATIVE = "negative"   # False positive - actually acceptable


@dataclass
class UnifiedTrainingExample:
    """
    Unified format for training examples from any source.

    This abstracts over:
    - Node-level human validation (direct examples)
    - Full-document labels (bootstrapped to node level)
    - Oracle approximation auto-reviews (filtered carefully)

    All training data sources convert to this format before being
    used for training.
    """
    # Identifiers
    example_id: str
    source_type: str  # "node_human", "document_label", "oracle_auto"

    # Input features (what the oracle sees)
    original_content: str
    summary: str
    rubric: str

    # Additional context
    context: Dict[str, Any] = field(default_factory=dict)

    # Labels
    label: TrainingExampleLabel = TrainingExampleLabel.POSITIVE
    violation_type: ViolationType = ViolationType.SUFFICIENCY

    # For positive examples: the corrected summary
    corrected_summary: Optional[str] = None
    human_reasoning: Optional[str] = None

    # Metadata
    confidence: float = 1.0  # Source reliability (human=1.0, auto=0.6-0.9)
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dspy_example(self) -> dspy.Example:
        """Convert to DSPy format for training."""
        is_violation = self.label == TrainingExampleLabel.POSITIVE

        # Get label for metric comparison:
        # - For RILE/ordinal: use discretized_label from context
        # - For categorical: use violation_type
        label = self.context.get('discretized_label')
        if label is None:
            label = self.violation_type.value if is_violation else "none"

        return dspy.Example(
            original_content=self.original_content,
            summary=self.summary,
            rubric=self.rubric,
            check_type=self.violation_type.value,
            is_true_violation=is_violation,
            violation_type=self.violation_type.value if is_violation else "none",
            label=str(label),  # For metric comparison
            confidence=self.confidence,
            corrected_summary=self.corrected_summary or "",
            reasoning=self.human_reasoning or "",
        ).with_inputs("original_content", "summary", "rubric")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'example_id': self.example_id,
            'source_type': self.source_type,
            'original_content': self.original_content,
            'summary': self.summary,
            'rubric': self.rubric,
            'context': self.context,
            'label': self.label.value,
            'violation_type': self.violation_type.value,
            'corrected_summary': self.corrected_summary,
            'human_reasoning': self.human_reasoning,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedTrainingExample':
        """Deserialize from dictionary."""
        data = dict(data)
        data['label'] = TrainingExampleLabel(data.get('label', 'positive'))
        data['violation_type'] = ViolationType(data.get('violation_type', 'sufficiency'))
        return cls(**data)


@runtime_checkable
class TrainingDataSource(Protocol):
    """
    Protocol for training data sources.

    Any class implementing this protocol can be used as a training
    data source for the oracle approximation framework.
    """

    def get_examples(self) -> List[UnifiedTrainingExample]:
        """Return all available training examples."""
        ...

    def get_positive_examples(self) -> List[UnifiedTrainingExample]:
        """Return positive (true violation) examples."""
        ...

    def get_negative_examples(self) -> List[UnifiedTrainingExample]:
        """Return negative (false positive) examples."""
        ...

    @property
    def source_type(self) -> str:
        """Identify the source type."""
        ...


@dataclass
class OracleReviewResult:
    """Result of reviewing a single item with the oracle."""
    item_id: str
    is_violation: bool
    violation_type: ViolationType
    confidence: float
    reasoning: str
    corrected_summary: Optional[str] = None
    candidates: List[str] = field(default_factory=list)

    @property
    def auto_decided(self) -> bool:
        """Whether confidence is high enough for automatic decision."""
        return self.confidence >= 0.8


# =============================================================================
# Label Space Abstraction
# =============================================================================

@runtime_checkable
class LabelSpace(Protocol):
    """
    Protocol for label spaces used in classification.

    Supports both categorical (unordered) and ordinal (ordered with distance)
    label spaces. This abstraction enables the same classifier to work with
    violation types (5 labels) and discretized RILE scores (41 labels).
    """

    @property
    def labels(self) -> List[str]:
        """Return all possible labels in this space."""
        ...

    @property
    def is_ordinal(self) -> bool:
        """Whether labels have a meaningful distance metric."""
        ...

    def distance(self, a: str, b: str) -> float:
        """
        Compute distance between two labels.

        For categorical: 0 if equal, 1 otherwise
        For ordinal: absolute difference between label values
        """
        ...

    def fits_in_context(self, max_labels: int = 10) -> bool:
        """Whether all labels can be enumerated in a prompt."""
        ...

    def get_description(self, label: str) -> str:
        """Get a text description of a label for embedding/retrieval."""
        ...


class CategoricalLabelSpace:
    """
    Label space for categorical (unordered) labels.

    Examples: ViolationType enum, binary classification, etc.
    """

    def __init__(
        self,
        labels: List[str],
        descriptions: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize categorical label space.

        Args:
            labels: List of label strings
            descriptions: Optional mapping of label -> description for retrieval
        """
        self._labels = labels
        self._descriptions = descriptions or {}

    @classmethod
    def from_enum(cls, enum_class: type, descriptions: Optional[Dict[str, str]] = None) -> 'CategoricalLabelSpace':
        """Create from an Enum class."""
        labels = [e.value for e in enum_class]
        return cls(labels, descriptions)

    @classmethod
    def from_violation_types(cls) -> 'CategoricalLabelSpace':
        """Create label space for OPS violation types with descriptions."""
        descriptions = {
            ViolationType.SUFFICIENCY.value: (
                "Sufficiency violation (C1): The leaf summary does not preserve "
                "enough information from the original content to compute the oracle. "
                "Key details relevant to the rubric are lost in summarization."
            ),
            ViolationType.MERGE_CONSISTENCY.value: (
                "Merge consistency violation (C3B): When merging child summaries, "
                "information relevant to the oracle was lost. The merged summary "
                "is not consistent with what the children summaries would predict."
            ),
            ViolationType.IDEMPOTENCE.value: (
                "Idempotence violation (C2): Re-summarizing the summary changes "
                "the oracle prediction. The summary is not stable under further "
                "summarization, indicating it contains extraneous detail."
            ),
            ViolationType.SUBSTITUTION.value: (
                "Substitution violation (C3A): Equivalent summaries at a boundary "
                "do not produce the same oracle prediction. The oracle is sensitive "
                "to surface-level differences that should not matter."
            ),
            ViolationType.NONE.value: (
                "No violation: The summarization is acceptable. The oracle prediction "
                "from the summary matches the prediction from the original content."
            ),
        }
        return cls.from_enum(ViolationType, descriptions)

    @property
    def labels(self) -> List[str]:
        return self._labels

    @property
    def is_ordinal(self) -> bool:
        return False

    def distance(self, a: str, b: str) -> float:
        """Categorical distance: 0 if equal, 1 otherwise."""
        return 0.0 if a == b else 1.0

    def fits_in_context(self, max_labels: int = 10) -> bool:
        return len(self._labels) <= max_labels

    def get_description(self, label: str) -> str:
        return self._descriptions.get(label, label)


class OrdinalLabelSpace:
    """
    Label space for ordinal (ordered) labels with a distance metric.

    Examples: Discretized RILE scores (-100 to +100 in bins of 5),
    Likert scales, confidence levels, etc.
    """

    def __init__(
        self,
        min_value: float,
        max_value: float,
        bin_size: float = 1.0,
        description_fn: Optional[callable] = None,
    ):
        """
        Initialize ordinal label space.

        Args:
            min_value: Minimum value in the range
            max_value: Maximum value in the range
            bin_size: Size of each bin (e.g., 5 for bins of 5)
            description_fn: Optional function(value) -> description string
        """
        self.min_value = min_value
        self.max_value = max_value
        self.bin_size = bin_size
        self._description_fn = description_fn

        # Generate bin centers as labels
        self._labels = []
        current = min_value
        while current <= max_value + 0.001:  # Small epsilon for float comparison
            self._labels.append(str(int(current) if bin_size >= 1 else current))
            current += bin_size

    @classmethod
    def for_rile(cls, bin_size: float = 5.0) -> 'OrdinalLabelSpace':
        """Create label space for RILE scores (-100 to +100)."""
        def rile_description(value: float) -> str:
            v = float(value)
            if v < -60:
                position = "Far left"
                emphasis = "strong welfare state, nationalization, internationalism"
            elif v < -20:
                position = "Left"
                emphasis = "welfare expansion, market regulation, social spending"
            elif v < 20:
                position = "Center"
                emphasis = "balanced policies, pragmatic approach"
            elif v < 60:
                position = "Right"
                emphasis = "free enterprise, limited government, traditional values"
            else:
                position = "Far right"
                emphasis = "strong free market, minimal state, nationalism"

            return f"RILE score {value}: {position} position. Policy emphasis: {emphasis}."

        return cls(
            min_value=-100,
            max_value=100,
            bin_size=bin_size,
            description_fn=rile_description,
        )

    @property
    def labels(self) -> List[str]:
        return self._labels

    @property
    def is_ordinal(self) -> bool:
        return True

    def distance(self, a: str, b: str) -> float:
        """Ordinal distance: absolute difference between values."""
        try:
            return abs(float(a) - float(b))
        except ValueError:
            return float('inf')

    def fits_in_context(self, max_labels: int = 10) -> bool:
        return len(self._labels) <= max_labels

    def get_description(self, label: str) -> str:
        if self._description_fn:
            return self._description_fn(float(label))
        return f"Value: {label}"

    def nearest_label(self, value: float) -> str:
        """Find the nearest label to a given value."""
        # Clamp to range
        value = max(self.min_value, min(self.max_value, value))
        # Find nearest bin
        bin_index = round((value - self.min_value) / self.bin_size)
        bin_value = self.min_value + bin_index * self.bin_size
        return str(int(bin_value) if self.bin_size >= 1 else bin_value)


# =============================================================================
# Prediction and Verification Results
# =============================================================================

@dataclass
class Prediction:
    """Result of classifying an item."""
    label: str
    confidence: float
    reasoning: str
    raw_scores: Optional[Dict[str, float]] = None  # Per-label scores if available

    def to_dict(self) -> Dict[str, Any]:
        return {
            'label': self.label,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'raw_scores': self.raw_scores,
        }


@dataclass
class LawCheckResult:
    """Result of checking an OPS law at a tree node."""
    law: str  # "sufficiency", "idempotence", "merge_consistency", "substitution"
    passed: bool
    discrepancy: float  # Distance between expected and actual (0 if passed)

    # Predictions that were compared
    original_prediction: Optional[Prediction] = None
    summary_prediction: Optional[Prediction] = None
    expected_label: Optional[str] = None  # For merge consistency

    # Context
    node_id: Optional[str] = None
    reasoning: Optional[str] = None

    def to_training_example(
        self,
        original_content: str,
        summary: str,
        rubric: str,
        example_id: str,
    ) -> UnifiedTrainingExample:
        """Convert law check result to training example."""
        return UnifiedTrainingExample(
            example_id=example_id,
            source_type="law_violation",
            original_content=original_content,
            summary=summary,
            rubric=rubric,
            label=TrainingExampleLabel.POSITIVE if not self.passed else TrainingExampleLabel.NEGATIVE,
            violation_type=ViolationType.from_check_type(self.law),
            context={
                'law': self.law,
                'discrepancy': self.discrepancy,
                'node_id': self.node_id,
            },
            human_reasoning=self.reasoning,
            confidence=0.8 if not self.passed else 0.9,  # Slightly lower confidence for violations
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'law': self.law,
            'passed': self.passed,
            'discrepancy': self.discrepancy,
            'original_prediction': self.original_prediction.to_dict() if self.original_prediction else None,
            'summary_prediction': self.summary_prediction.to_dict() if self.summary_prediction else None,
            'expected_label': self.expected_label,
            'node_id': self.node_id,
            'reasoning': self.reasoning,
        }
