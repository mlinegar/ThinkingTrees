"""
Domain plugin base protocol and abstractions.

This module defines the interface for domain-specific training integration.
Domains represent different use cases (e.g., manifesto/RILE, legal documents,
sentiment analysis) that can plug into the OPS training framework.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    import dspy
    from ..core import LabelSpace, TrainingDataSource
    from ..config import OracleIRRConfig
    from ..inference import Retriever

logger = logging.getLogger(__name__)


# =============================================================================
# Domain Plugin Protocol
# =============================================================================

@runtime_checkable
class DomainPlugin(Protocol):
    """
    Protocol for domain-specific training integration.

    A domain plugin provides all the domain-specific components needed to
    train and evaluate oracle classifiers for a particular use case.

    Example domains:
    - manifesto_rile: Political manifesto RILE scoring
    - legal_relevance: Legal document relevance scoring
    - sentiment: Sentiment analysis classification
    """

    @property
    def name(self) -> str:
        """Unique identifier for this domain."""
        ...

    @property
    def label_space(self) -> 'LabelSpace':
        """Label space for this domain (ordinal or categorical)."""
        ...

    def create_training_source(
        self,
        results: List[Any],
        **kwargs,
    ) -> 'TrainingDataSource':
        """
        Create a training data source from processing results.

        Args:
            results: List of domain-specific result objects
            **kwargs: Domain-specific configuration

        Returns:
            TrainingDataSource that can be added to UnifiedTrainingCollector
        """
        ...

    def create_metric(
        self,
        weighted: bool = True,
        with_feedback: bool = True,
    ) -> Callable:
        """
        Create a DSPy-compatible metric function for this domain.

        Args:
            weighted: Whether to use weighted scoring (for ordinal spaces)
            with_feedback: Whether to include feedback for GEPA reflection

        Returns:
            Metric function compatible with DSPy optimizers
        """
        ...

    def create_classifier(
        self,
        retriever: Optional['Retriever'] = None,
        config: Optional['OracleIRRConfig'] = None,
    ) -> 'dspy.Module':
        """
        Create a classifier module for this domain.

        Args:
            retriever: Optional retriever for retrieval-augmented classification
            config: Optional Oracle IRR configuration

        Returns:
            DSPy Module for classification
        """
        ...

    def create_rubric(self, **kwargs) -> str:
        """
        Create a rubric string for this domain.

        Rubrics guide the summarization and classification process.

        Args:
            **kwargs: Domain-specific rubric configuration

        Returns:
            Rubric string
        """
        ...


# =============================================================================
# Abstract Base Class
# =============================================================================

class AbstractDomain(ABC):
    """
    Abstract base class for domain implementations.

    Provides common functionality and enforces the interface.
    Subclasses should implement the abstract methods.
    """

    def __init__(self):
        """Initialize domain."""
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this domain."""
        pass

    @property
    @abstractmethod
    def label_space(self) -> 'LabelSpace':
        """Label space for this domain."""
        pass

    @abstractmethod
    def create_training_source(
        self,
        results: List[Any],
        **kwargs,
    ) -> 'TrainingDataSource':
        """Create a training data source from processing results."""
        pass

    @abstractmethod
    def create_metric(
        self,
        weighted: bool = True,
        with_feedback: bool = True,
    ) -> Callable:
        """Create a DSPy-compatible metric function."""
        pass

    @abstractmethod
    def create_classifier(
        self,
        retriever: Optional['Retriever'] = None,
        config: Optional['OracleIRRConfig'] = None,
    ) -> 'dspy.Module':
        """Create a classifier module for this domain."""
        pass

    @abstractmethod
    def create_rubric(self, **kwargs) -> str:
        """Create a rubric string for this domain."""
        pass

    def validate(self) -> bool:
        """
        Validate that the domain is properly configured.

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required properties
            _ = self.name
            _ = self.label_space
            return True
        except Exception as e:
            logger.error(f"Domain validation failed: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this domain.

        Returns:
            Dict with domain metadata
        """
        return {
            'name': self.name,
            'label_space_type': type(self.label_space).__name__,
            'is_ordinal': getattr(self.label_space, 'is_ordinal', False),
        }
