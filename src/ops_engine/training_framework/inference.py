"""
Oracle Classifier with Retrieval Augmentation.

This module implements the core classification functionality for the oracle
approximation framework, supporting both categorical and ordinal label spaces
with configurable retrieval strategies.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import dspy
import torch

from .core import (
    LabelSpace,
    CategoricalLabelSpace,
    OrdinalLabelSpace,
    Prediction,
    UnifiedTrainingExample,
)
from .config import OracleIRRConfig


# =============================================================================
# Retriever
# =============================================================================

class Retriever:
    """
    Semantic retriever using SentenceTransformer embeddings.

    Supports two modes:
    - Label retrieval: Find top-k candidate labels from a large label space
    - Example retrieval: Find similar training examples for few-shot context
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the retriever.

        Args:
            model_name: SentenceTransformer model name
            cache_dir: Directory for caching embeddings
            device: Device to use ("cpu", "cuda", or None for auto)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or Path("data/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy load model
        self._model = None
        self._device = device

        # Caches for label and example embeddings
        self._label_embeddings: Optional[torch.Tensor] = None
        self._label_texts: List[str] = []
        self._example_embeddings: Optional[torch.Tensor] = None
        self._examples: List[UnifiedTrainingExample] = []

    @property
    def model(self):
        """Lazy load the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            if self._device:
                self._model.to(self._device)
            else:
                self._model.to("cpu")
        return self._model

    def index_labels(self, label_space: LabelSpace, force_recompute: bool = False):
        """
        Index labels from a label space for retrieval.

        Args:
            label_space: The label space to index
            force_recompute: Whether to recompute embeddings even if cached
        """
        # Get label descriptions for embedding
        self._label_texts = [
            label_space.get_description(label)
            for label in label_space.labels
        ]

        # Check cache
        cache_key = f"labels_{hash(tuple(self._label_texts))}"
        cache_path = self.cache_dir / f"{cache_key}.pt"

        if cache_path.exists() and not force_recompute:
            self._label_embeddings = torch.load(cache_path, map_location="cpu")
        else:
            # Compute embeddings
            self._label_embeddings = self.model.encode(
                self._label_texts,
                convert_to_tensor=True,
                show_progress_bar=len(self._label_texts) > 100,
            )
            torch.save(self._label_embeddings, cache_path)

    def index_examples(
        self,
        examples: List[UnifiedTrainingExample],
        force_recompute: bool = False,
    ):
        """
        Index training examples for retrieval.

        Args:
            examples: List of training examples to index
            force_recompute: Whether to recompute embeddings even if cached
        """
        self._examples = examples

        # Create text representation of each example
        example_texts = [
            f"Content: {ex.original_content[:500]}... Summary: {ex.summary[:300]}..."
            for ex in examples
        ]

        # Compute embeddings
        self._example_embeddings = self.model.encode(
            example_texts,
            convert_to_tensor=True,
            show_progress_bar=len(example_texts) > 100,
        )

    def retrieve_labels(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[float, str]]:
        """
        Retrieve top-k labels most similar to the query.

        Args:
            query: Query text to match against labels
            top_k: Number of labels to retrieve

        Returns:
            List of (score, label_description) tuples, sorted by score descending
        """
        if self._label_embeddings is None:
            raise ValueError("Labels not indexed. Call index_labels() first.")

        from sentence_transformers import util as st_util

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = st_util.cos_sim(query_embedding, self._label_embeddings)[0]

        # Get top-k indices
        top_indices = torch.topk(scores, min(top_k, len(self._label_texts))).indices

        return [
            (scores[i].item(), self._label_texts[i])
            for i in top_indices
        ]

    def retrieve_examples(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Tuple[float, UnifiedTrainingExample]]:
        """
        Retrieve top-k training examples most similar to the query.

        Args:
            query: Query text to match against examples
            top_k: Number of examples to retrieve

        Returns:
            List of (score, example) tuples, sorted by score descending
        """
        if self._example_embeddings is None:
            raise ValueError("Examples not indexed. Call index_examples() first.")

        from sentence_transformers import util as st_util

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = st_util.cos_sim(query_embedding, self._example_embeddings)[0]

        # Get top-k indices
        top_indices = torch.topk(scores, min(top_k, len(self._examples))).indices

        return [
            (scores[i].item(), self._examples[i])
            for i in top_indices
        ]


# =============================================================================
# DSPy Signatures
# =============================================================================

class OracleClassifySignature(dspy.Signature):
    """
    Classify text into a label from the given candidates.

    Given original content, its summary, and a rubric describing what to preserve,
    determine which label best describes the content/summary pair.
    """
    original_content: str = dspy.InputField(
        desc="The original text content before summarization"
    )
    summary: str = dspy.InputField(
        desc="The summary of the original content"
    )
    rubric: str = dspy.InputField(
        desc="Description of what information should be preserved in the summary"
    )
    candidate_labels: str = dspy.InputField(
        desc="Candidate labels to choose from, with descriptions"
    )
    few_shot_examples: str = dspy.InputField(
        desc="Similar examples for reference (may be empty)"
    )

    reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning about why this label applies"
    )
    label: str = dspy.OutputField(
        desc="The predicted label (must be one of the candidates)"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score from 0.0 to 1.0"
    )


class OracleViolationSignature(dspy.Signature):
    """
    Determine if a summary violates OPS laws by losing oracle-relevant information.

    Compare the original content and summary to determine:
    1. Whether the summary preserves enough information to compute the oracle
    2. What type of violation occurred (if any)
    3. How confident we are in this assessment
    """
    original_content: str = dspy.InputField(
        desc="The original text content before summarization"
    )
    summary: str = dspy.InputField(
        desc="The summary of the original content"
    )
    rubric: str = dspy.InputField(
        desc="Description of what information should be preserved (oracle criteria)"
    )
    violation_types: str = dspy.InputField(
        desc="Possible violation types with descriptions"
    )

    is_violation: bool = dspy.OutputField(
        desc="Whether the summary violates OPS laws (loses oracle-relevant info)"
    )
    violation_type: str = dspy.OutputField(
        desc="Type of violation if is_violation=True, else 'none'"
    )
    reasoning: str = dspy.OutputField(
        desc="Explanation of why this is/isn't a violation"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score from 0.0 to 1.0"
    )


# =============================================================================
# Oracle Classifier
# =============================================================================

class OracleClassifier(dspy.Module):
    """
    Classifies items into a label space with optional retrieval augmentation.

    Supports two modes based on label space size:
    - Small spaces (≤10 labels): Enumerate all labels, retrieve similar examples
    - Large spaces (>10 labels): Retrieve top-k candidate labels

    This classifier is the core component used by OracleNodeVerifier to check
    OPS law compliance at each node in the summarization tree.
    """

    def __init__(
        self,
        label_space: LabelSpace,
        retriever: Optional[Retriever] = None,
        config: Optional[OracleIRRConfig] = None,
    ):
        """
        Initialize the classifier.

        Args:
            label_space: The label space to classify into
            retriever: Optional retriever for augmentation
            config: Configuration options
        """
        super().__init__()

        self.label_space = label_space
        self.retriever = retriever
        self.config = config or OracleIRRConfig()

        # Determine retrieval strategy
        self.use_label_retrieval = not label_space.fits_in_context(
            self.config.rank_topk
        )

        # DSPy modules
        self.classify = dspy.ChainOfThought(OracleClassifySignature)

        # Index labels if using label retrieval
        if self.use_label_retrieval and self.retriever:
            self.retriever.index_labels(label_space)

    def set_training_examples(self, examples: List[UnifiedTrainingExample]):
        """Set training examples for few-shot retrieval."""
        if self.retriever and not self.use_label_retrieval:
            self.retriever.index_examples(examples)

    def forward(
        self,
        original_content: str,
        summary: str,
        rubric: str,
        context: Optional[Dict] = None,
        **kwargs,  # Accept extra fields from DSPy Examples
    ) -> Prediction:
        """
        Classify the content/summary pair.

        Args:
            original_content: Original text before summarization
            summary: The summary to evaluate
            rubric: Description of what to preserve
            context: Optional additional context
            **kwargs: Extra fields (ignored, allows DSPy Example compatibility)

        Returns:
            Prediction with label, confidence, and reasoning
        """
        # Build query for retrieval
        query = f"Content: {original_content[:500]} Summary: {summary[:300]} Rubric: {rubric}"

        # Get candidates and examples based on strategy
        if self.use_label_retrieval and self.retriever:
            # Large space: retrieve top-k labels
            retrieved = self.retriever.retrieve_labels(query, top_k=self.config.rank_topk)
            candidate_labels = "\n".join([
                f"- {desc} (score: {score:.3f})"
                for score, desc in retrieved
            ])
            few_shot = ""
        else:
            # Small space: enumerate all labels
            candidate_labels = "\n".join([
                f"- {label}: {self.label_space.get_description(label)}"
                for label in self.label_space.labels
            ])

            # Retrieve similar examples if available
            few_shot = ""
            if self.retriever and self.retriever._example_embeddings is not None:
                retrieved_examples = self.retriever.retrieve_examples(query, top_k=3)
                if retrieved_examples:
                    few_shot = "Similar examples:\n"
                    for score, ex in retrieved_examples:
                        few_shot += f"\n---\nContent: {ex.original_content[:200]}...\n"
                        few_shot += f"Summary: {ex.summary[:150]}...\n"
                        few_shot += f"Label: {ex.violation_type.value}\n"
                        if ex.human_reasoning:
                            few_shot += f"Reasoning: {ex.human_reasoning}\n"

        # Run classification
        result = self.classify(
            original_content=original_content,
            summary=summary,
            rubric=rubric,
            candidate_labels=candidate_labels,
            few_shot_examples=few_shot or "None provided",
        )

        # Parse and validate result
        label = self._parse_label(result.label)
        confidence = self._parse_confidence(result.confidence)

        return Prediction(
            label=label,
            confidence=confidence,
            reasoning=result.reasoning,
        )

    def _parse_label(self, label_str: str) -> str:
        """Parse and validate the predicted label."""
        label_str = label_str.strip().lower()

        # Try exact match first
        for label in self.label_space.labels:
            if label.lower() == label_str:
                return label

        # Try partial match
        for label in self.label_space.labels:
            if label.lower() in label_str or label_str in label.lower():
                return label

        # For ordinal, try to extract a number
        if self.label_space.is_ordinal:
            import re
            numbers = re.findall(r'-?\d+\.?\d*', label_str)
            if numbers:
                value = float(numbers[0])
                if isinstance(self.label_space, OrdinalLabelSpace):
                    return self.label_space.nearest_label(value)

        # Default to first label
        return self.label_space.labels[0]

    def _parse_confidence(self, conf_val) -> float:
        """Parse confidence value."""
        try:
            if isinstance(conf_val, (int, float)):
                conf = float(conf_val)
            else:
                import re
                numbers = re.findall(r'\d+\.?\d*', str(conf_val))
                conf = float(numbers[0]) if numbers else 0.5
            return max(0.0, min(1.0, conf))
        except (ValueError, IndexError):
            return 0.5


# =============================================================================
# Specialized Classifiers
# =============================================================================

class ViolationClassifier(OracleClassifier):
    """
    Specialized classifier for OPS law violations.

    Uses the ViolationType label space and OracleViolationSignature
    for more targeted violation detection.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        config: Optional[OracleIRRConfig] = None,
    ):
        label_space = CategoricalLabelSpace.from_violation_types()
        super().__init__(label_space, retriever, config)

        # Use specialized signature
        self.detect_violation = dspy.ChainOfThought(OracleViolationSignature)

    def forward(
        self,
        original_content: str,
        summary: str,
        rubric: str,
        context: Optional[Dict] = None,
        **kwargs,  # Accept extra fields from DSPy Examples
    ) -> Prediction:
        """Detect violations with specialized signature."""
        # Build violation types description
        violation_types = "\n".join([
            f"- {label}: {self.label_space.get_description(label)}"
            for label in self.label_space.labels
        ])

        # Run detection
        result = self.detect_violation(
            original_content=original_content,
            summary=summary,
            rubric=rubric,
            violation_types=violation_types,
        )

        # Parse result
        is_violation = self._parse_bool(result.is_violation)
        label = result.violation_type if is_violation else "none"
        label = self._parse_label(label)
        confidence = self._parse_confidence(result.confidence)

        return Prediction(
            label=label,
            confidence=confidence,
            reasoning=result.reasoning,
        )

    def _parse_bool(self, val) -> bool:
        """Parse boolean value."""
        if isinstance(val, bool):
            return val
        val_str = str(val).lower().strip()
        return val_str in ('true', 'yes', '1', 'violation', 'violated')


class RILEClassifier(OracleClassifier):
    """
    Specialized classifier for RILE score prediction.

    Uses ordinal label space with discretized RILE scores.
    """

    def __init__(
        self,
        bin_size: float = 5.0,
        retriever: Optional[Retriever] = None,
        config: Optional[OracleIRRConfig] = None,
    ):
        label_space = OrdinalLabelSpace.for_rile(bin_size=bin_size)
        super().__init__(label_space, retriever, config)

    def predict_rile(
        self,
        content: str,
        rubric: str = "Political ideology indicators: left-wing (welfare, regulation, internationalism) vs right-wing (free market, nationalism, traditional values)",
    ) -> Tuple[float, float, str]:
        """
        Predict RILE score for content.

        Returns:
            Tuple of (predicted_rile, confidence, reasoning)
        """
        # For RILE, we classify the content directly (not original vs summary)
        pred = self(
            original_content=content,
            summary=content,  # Same for direct classification
            rubric=rubric,
        )

        try:
            rile_value = float(pred.label)
        except ValueError:
            rile_value = 0.0

        return rile_value, pred.confidence, pred.reasoning
