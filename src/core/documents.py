"""
Document-level data models.

These models are dataset-agnostic and keep document metadata in a structured form.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DocumentSample:
    """A single document sample."""
    doc_id: str
    text: str
    reference_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like access for metadata."""
        return self.metadata.get(key, default)

    @property
    def manifesto_id(self) -> str:
        """Legacy alias for doc_id."""
        return self.doc_id


@dataclass
class DocumentResult:
    """Result of processing a single document."""
    doc_id: str
    reference_score: Optional[float] = None
    estimated_score: Optional[float] = None
    baseline_score: Optional[float] = None

    final_summary: str = ""
    summary_length: int = 0
    original_length: int = 0
    compression_ratio: float = 1.0

    tree_height: Optional[int] = None
    tree_nodes: Optional[int] = None
    tree_leaves: Optional[int] = None

    chunks: list = field(default_factory=list)
    leaf_summaries: list = field(default_factory=list)
    level_history: list = field(default_factory=list)
    processing_time: float = 0.0

    error: Optional[str] = None
    reasoning: str = ""
    left_indicators: str = ""
    right_indicators: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def prediction_error(self) -> Optional[float]:
        if self.estimated_score is None or self.reference_score is None:
            return None
        return abs(self.estimated_score - self.reference_score)

    @property
    def baseline_error(self) -> Optional[float]:
        if self.baseline_score is None or self.reference_score is None:
            return None
        return abs(self.baseline_score - self.reference_score)

    @property
    def predicted_rile(self) -> Optional[float]:
        """Legacy alias for estimated_score."""
        return self.estimated_score

    @property
    def ground_truth_rile(self) -> Optional[float]:
        """Legacy alias for reference_score."""
        return self.reference_score
