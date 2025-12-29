"""
Evaluation Metrics for Oracle Approximation.

This module provides metrics for evaluating:
- Continuous score prediction (using BoundedScale)
- OPS law compliance rates
- DSPy-compatible metric functions

New API (Preferred):
    from src.ops_engine.scoring import oracle_as_metric, ScoringOracle

    # ScoringOracle -> DSPy metric in one line
    metric = oracle_as_metric(my_scorer)
    optimizer.compile(student, trainset, metric=metric)

Summarization metrics:
    from src.ops_engine.training_framework.metrics import summarization

    # When you need specialized summarization metrics with quality checks
    metric = summarization(oracle_classifier, ...)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable
import math
import warnings

from .core import Prediction, LawCheckResult

# Re-export score-centric metric converters for convenience
from src.ops_engine.scoring import (
    oracle_as_metric,
    oracle_as_metric_with_feedback,
    ScoringOracle,
    OracleScore,
    normalize_error_to_score,
    BoundedScale,
    PERCENT_SCALE,
    Oracle,
    OraclePrediction,
)


# =============================================================================
# Generic Metric Factory (Scale-Based)
# =============================================================================

def metric(
    oracle_fn: Callable,
    scale: BoundedScale,
    ground_truth_field: str = 'ground_truth',
    prediction_field: str = 'label',
    with_feedback: bool = False,
) -> Callable:
    """
    Create a DSPy metric returning a normalized 0-1 score.

    Score computation (euclidean distance normalized by scale range):
        score = 1 - abs(predicted - ground_truth) / scale.range

    For RILE (-100 to 100, range=200):
        score = 1 - abs(pred - gt) / 200

    Examples:
        - pred=50, gt=50   → score = 1.0 (perfect match)
        - pred=50, gt=-50  → score = 1 - 100/200 = 0.5
        - pred=100, gt=-100 → score = 1 - 200/200 = 0.0 (max error)

    Args:
        oracle_fn: Function that scores text, returns value on the scale.
            Signature: (text: str) -> float  OR
            Signature: (text: str) -> Tuple[float, float, str] (value, confidence, reasoning)
        scale: BoundedScale defining the value range for normalization
        ground_truth_field: Field name for ground truth on gold example
        prediction_field: Field name for predicted value on prediction
        with_feedback: Return dict with feedback for GEPA

    Returns:
        DSPy-compatible metric function returning pure score (0-1).
        Use create_combined_metric() to combine multiple metrics with weights.

    Example:
        # RILE political positioning metric
        rile_scale = BoundedScale(-100.0, 100.0)
        metric = create_metric(
            oracle_fn=rile_oracle.predict,
            scale=rile_scale,
        )
    """
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        # Get ground truth
        ground_truth = getattr(gold, ground_truth_field, 0.0)

        # Get prediction - handle both string returns and object returns
        if isinstance(pred, str):
            text_to_score = pred
        else:
            text_to_score = getattr(pred, 'summary', None)
            if text_to_score is None:
                text_to_score = getattr(pred, prediction_field, None)
            if text_to_score is None:
                text_to_score = str(pred)

        # Call oracle function
        try:
            result = oracle_fn(text_to_score)
            # Handle different return types
            if isinstance(result, tuple):
                predicted_value = result[0]  # (value, confidence, reasoning)
                reasoning = result[2] if len(result) > 2 else ""
            else:
                predicted_value = float(result)
                reasoning = ""
        except Exception as e:
            if with_feedback:
                return {'score': 0.0, 'feedback': f"Oracle error: {str(e)[:50]}"}
            return 0.0

        # Compute score using scale
        score = scale.values_to_score(predicted_value, ground_truth)

        if with_feedback:
            feedback = f"Predicted {predicted_value:.1f}, expected {ground_truth:.1f}"
            if reasoning:
                feedback += f" ({reasoning})"
            return {'score': score, 'feedback': feedback}

        return score

    return metric


# Note: cached_oracle has been removed.
# Use @functools.lru_cache decorator directly on oracle prediction functions instead.

# Note: path_aggregate_score and probabilistic_audit have been removed.
# These were complex tree-path-based metrics that are no longer used.
# Use oracle_score_prediction for simpler oracle-based scoring.

# =============================================================================
# Calibration Metrics
# =============================================================================

def calibration_error(
    predictions: List[Prediction],
    ground_truth: List[str],
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute calibration error metrics.

    A well-calibrated classifier should have predictions with confidence X
    being correct X% of the time.

    Args:
        predictions: List of Prediction objects with confidence scores
        ground_truth: List of true labels
        n_bins: Number of confidence bins

    Returns:
        Dict with ECE (expected calibration error) and MCE (max calibration error)
    """
    if not predictions or len(predictions) != len(ground_truth):
        return {'ece': 0.0, 'mce': 0.0}

    # Bin predictions by confidence
    bins = [[] for _ in range(n_bins)]
    for pred, true in zip(predictions, ground_truth):
        bin_idx = min(int(pred.confidence * n_bins), n_bins - 1)
        correct = 1 if pred.label == true else 0
        bins[bin_idx].append((pred.confidence, correct))

    # Compute per-bin accuracy and confidence
    ece = 0.0
    mce = 0.0
    total = len(predictions)

    for bin_data in bins:
        if not bin_data:
            continue

        n = len(bin_data)
        avg_confidence = sum(c for c, _ in bin_data) / n
        accuracy = sum(corr for _, corr in bin_data) / n
        gap = abs(accuracy - avg_confidence)

        # Weighted contribution to ECE
        ece += (n / total) * gap
        mce = max(mce, gap)

    return {'ece': ece, 'mce': mce}


def reliability_diagram_data(
    predictions: List[Prediction],
    ground_truth: List[str],
    n_bins: int = 10,
) -> List[Dict]:
    """
    Compute data for a reliability diagram.

    Args:
        predictions: List of Prediction objects
        ground_truth: List of true labels
        n_bins: Number of confidence bins

    Returns:
        List of dicts with bin_center, accuracy, confidence, and count
    """
    if not predictions:
        return []

    # Bin predictions
    bins = [[] for _ in range(n_bins)]
    for pred, true in zip(predictions, ground_truth):
        bin_idx = min(int(pred.confidence * n_bins), n_bins - 1)
        correct = 1 if pred.label == true else 0
        bins[bin_idx].append((pred.confidence, correct))

    # Compute per-bin stats
    data = []
    for i, bin_data in enumerate(bins):
        bin_center = (i + 0.5) / n_bins
        if bin_data:
            accuracy = sum(corr for _, corr in bin_data) / len(bin_data)
            confidence = sum(c for c, _ in bin_data) / len(bin_data)
            count = len(bin_data)
        else:
            accuracy = None
            confidence = None
            count = 0

        data.append({
            'bin_center': bin_center,
            'accuracy': accuracy,
            'confidence': confidence,
            'count': count,
        })

    return data


# =============================================================================
# OPS Law Compliance Metrics
# =============================================================================

def law_compliance_rate(
    check_results: List[LawCheckResult],
) -> Dict[str, float]:
    """
    Compute compliance rate for each OPS law.

    Args:
        check_results: List of LawCheckResult objects

    Returns:
        Dict mapping law name to pass rate [0, 1]
    """
    by_law: Dict[str, List[bool]] = {}

    for result in check_results:
        if result.law not in by_law:
            by_law[result.law] = []
        by_law[result.law].append(result.passed)

    return {
        law: sum(passes) / len(passes) if passes else 1.0
        for law, passes in by_law.items()
    }


def overall_compliance_rate(check_results: List[LawCheckResult]) -> float:
    """
    Compute overall OPS law compliance rate.

    Args:
        check_results: List of LawCheckResult objects

    Returns:
        Overall pass rate [0, 1]
    """
    if not check_results:
        return 1.0

    passed = sum(r.passed for r in check_results)
    return passed / len(check_results)


def average_discrepancy(
    check_results: List[LawCheckResult],
) -> Dict[str, float]:
    """
    Compute average discrepancy for each OPS law.

    Args:
        check_results: List of LawCheckResult objects

    Returns:
        Dict mapping law name to average discrepancy
    """
    by_law: Dict[str, List[float]] = {}

    for result in check_results:
        if result.law not in by_law:
            by_law[result.law] = []
        by_law[result.law].append(result.discrepancy)

    return {
        law: sum(discs) / len(discs) if discs else 0.0
        for law, discs in by_law.items()
    }


# =============================================================================
# DSPy Metric Functions
# =============================================================================

# Note: classification() and violation() have been removed.
# Use continuous score prediction with oracle_as_metric() or metric() instead.
# Use compliance() for law compliance metrics.


def compliance(
    tolerance: float = 0.0,
) -> Callable:
    """
    Create a DSPy metric based on OPS law compliance.

    Args:
        tolerance: Allowed discrepancy before counting as failure

    Returns:
        Metric function
    """
    def metric(example, prediction, trace=None, pred_name=None, pred_trace=None) -> float:
        """DSPy metric for law compliance. Compatible with GEPA's 5-argument signature."""
        # Get discrepancy from prediction if available
        discrepancy = getattr(prediction, 'discrepancy', None)
        if discrepancy is not None:
            return 1.0 if discrepancy <= tolerance else 0.0

        # Otherwise, compare labels directly
        true_label = getattr(example, 'label', None)
        pred_label = getattr(prediction, 'label', None)

        if true_label is None or pred_label is None:
            return 0.0

        return 1.0 if true_label == pred_label else 0.0

    return metric


# =============================================================================
# Advanced DSPy Metrics (LLM Judge)
# =============================================================================

# Note: classification_trace() has been removed. Use continuous score prediction
# with oracle_as_metric() or metric() instead.


def llm_judge(judge_lm=None) -> Callable:
    """
    Create a metric that uses an LLM as judge for nuanced evaluation.

    This follows DSPy best practices for using DSPy programs as metrics.
    Useful when:
    - Simple distance metrics miss important nuances
    - Need to evaluate reasoning quality
    - Want domain-expert-like feedback

    Args:
        judge_lm: Optional LM to use for judging (uses default if None)

    Returns:
        Metric function that uses LLM as judge
    """
    try:
        import dspy
        from contextlib import nullcontext
    except ImportError:
        # Return a fallback metric if DSPy not available
        def fallback_metric(gold, pred, trace=None):
            return 0.5  # Neutral score
        return fallback_metric

    # Define the judge signature
    class MetricJudge(dspy.Signature):
        """Evaluate the quality of a political classification prediction."""

        original_text: str = dspy.InputField(desc="The original manifesto text (truncated)")
        summary: str = dspy.InputField(desc="The summary being classified")
        predicted_label: str = dspy.InputField(desc="The predicted RILE label")
        predicted_reasoning: str = dspy.InputField(desc="The model's reasoning for the prediction")
        true_label: str = dspy.InputField(desc="The ground truth label")

        score: float = dspy.OutputField(desc="Score from 0.0 to 1.0 based on prediction quality")
        feedback: str = dspy.OutputField(desc="Specific feedback for improvement")

    # Create the judge module
    judge = dspy.ChainOfThought(MetricJudge)

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        """
        LLM-as-judge metric for nuanced evaluation.

        Compatible with GEPA's 5-argument signature.

        Args:
            gold: Ground truth example
            pred: Prediction
            trace: Optional trace (not used by this metric)
            pred_name: Optional predictor name (for GEPA)
            pred_trace: Optional predictor trace (for GEPA)

        Returns:
            {'score': float, 'feedback': str}
        """
        # Get prediction details
        original_text = getattr(gold, 'original_content', '')
        if not original_text:
            original_text = getattr(gold, 'text', '')
        original_text = original_text[:2000]  # Truncate for context limits

        summary = getattr(gold, 'summary', '')
        pred_label = str(getattr(pred, 'label', ''))
        pred_reasoning = getattr(pred, 'reasoning', '')
        true_label = str(getattr(gold, 'label', ''))

        if not true_label:
            return {'score': 0.0, 'feedback': 'No ground truth label.'}

        try:
            # Use judge LM if different from main LM
            context = dspy.context(lm=judge_lm) if judge_lm else nullcontext()

            with context:
                result = judge(
                    original_text=original_text,
                    summary=summary,
                    predicted_label=pred_label,
                    predicted_reasoning=pred_reasoning,
                    true_label=true_label,
                )

            # Parse score
            try:
                score = float(result.score)
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            except (ValueError, TypeError, AttributeError):
                score = 0.5  # Default if parsing fails

            feedback = getattr(result, 'feedback', 'No feedback provided.')

            return {'score': score, 'feedback': feedback}

        except Exception as e:
            # Fallback on any error
            return {
                'score': 0.5,
                'feedback': f'Judge evaluation failed: {str(e)[:100]}'
            }

    return metric


# Note: composite() has been removed. Use combine_metrics() or combine_feedback()
# with oracle_as_metric() for composing metrics.


# =============================================================================
# Summarization Metrics (for Two-Step Iterative Optimization)
# =============================================================================

def summarization(
    oracle_classifier,
    human_weight: float = 0.3,
    oracle_weight: float = 0.7,
    threshold: float = 10.0,
    min_summary_length: int = 50,
    max_error: float = 100.0,
) -> Callable:
    """
    Create a metric for evaluating summary quality against score preservation.

    This metric is used to optimize summarization prompts in the two-step
    iterative process:
    1. Train oracle classifier on current summaries
    2. Optimize summarizers using oracle + human feedback as metric

    The metric combines:
    - Oracle score: Does the summary preserve the target score?
    - Human feedback: Historical corrections from review queue
    - Quality checks: Summary length, compression, clarity

    Args:
        oracle_classifier: Trained classifier with predict_rile(text) method
            that returns (score, confidence, reasoning)
        human_weight: Weight for human feedback score (0.0-1.0)
        oracle_weight: Weight for oracle-based score (should sum to 1.0 with human_weight)
        threshold: Maximum acceptable score drift
        min_summary_length: Minimum acceptable summary length
        max_error: Scale for error normalization (default 100.0).
            For RILE scores, use RILE_RANGE (200.0) from src.manifesto.constants.

    Returns:
        Metric function compatible with DSPy optimizers (including GEPA)

    Example:
        # Generic usage with 100-point scale
        metric = create_summarization_metric(
            oracle_classifier=oracle,
            max_error=100.0,
        )

        # RILE-specific usage (see src.manifesto.examples for full patterns)
        from src.manifesto.constants import RILE_RANGE
        metric = create_summarization_metric(
            oracle_classifier=oracle,
            max_error=RILE_RANGE,  # 200.0
        )
    """

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        """
        Evaluate summary quality for RILE preservation.

        Compatible with GEPA's 5-argument signature.

        Args:
            gold: Training example with:
                - original_text: The original chunk/text being summarized
                - ground_truth_rile: The true RILE score for the full document
                - rubric: The preservation rubric
                - human_score: Optional human feedback score (0.0-1.0)
            pred: Prediction with:
                - summary: The generated summary
            trace: Optional DSPy trace (for debugging)
            pred_name: Optional predictor name (for GEPA)
            pred_trace: Optional predictor trace (for GEPA)

        Returns:
            {'score': float, 'feedback': str} for GEPA compatibility
        """
        # Extract inputs
        original_text = getattr(gold, 'original_text', '')
        if not original_text:
            original_text = getattr(gold, 'content', '')

        ground_truth_rile = getattr(gold, 'ground_truth_rile', 0.0)
        human_score = getattr(gold, 'human_score', None)

        # Extract prediction - handle both string returns and object returns
        if isinstance(pred, str):
            summary = pred
        else:
            summary = getattr(pred, 'summary', '')
            if not summary:
                summary = getattr(pred, 'merged_summary', '')
            if not summary:
                summary = str(pred)

        # Initialize feedback
        feedback_parts = []
        scores = {}

        # 1. Oracle score: Does summary preserve RILE positioning?
        try:
            oracle_pred_rile, confidence, _ = oracle_classifier.predict_rile(summary)
            rile_diff = abs(oracle_pred_rile - ground_truth_rile)

            # Normalize to 0-1: perfect preservation = 1.0, max_error points off = 0.0
            oracle_score = normalize_error_to_score(rile_diff, max_error=max_error)
            scores['oracle'] = oracle_score

            if rile_diff > threshold:
                feedback_parts.append(
                    f"RILE drift detected: oracle predicted {oracle_pred_rile:.1f}, "
                    f"expected ~{ground_truth_rile:.1f} (diff={rile_diff:.1f}). "
                    f"Preserve more politically relevant content."
                )
            if confidence < 0.5:
                feedback_parts.append(
                    "Oracle uncertain about political positioning - "
                    "summary may be too vague or miss key indicators."
                )
        except Exception as e:
            oracle_score = 0.5  # Default on error
            scores['oracle'] = oracle_score
            feedback_parts.append(f"Oracle evaluation failed: {str(e)[:50]}")

        # 2. Quality checks
        quality_penalty = 0.0

        # Check summary length
        if len(summary) < min_summary_length:
            quality_penalty += 0.1
            feedback_parts.append(
                f"Summary too short ({len(summary)} chars < {min_summary_length}). "
                "May have lost important information."
            )

        # Check compression ratio
        if original_text:
            compression = len(summary) / max(len(original_text), 1)
            if compression < 0.05:  # Less than 5% of original
                quality_penalty += 0.05
                feedback_parts.append(
                    f"Extreme compression ({compression:.1%}) - may lose detail."
                )
            elif compression > 0.8:  # More than 80% of original
                quality_penalty += 0.05
                feedback_parts.append(
                    f"Low compression ({compression:.1%}) - summary should be more concise."
                )

        # Check for empty or minimal content
        if not summary.strip():
            quality_penalty = 0.5
            feedback_parts.append("Summary is empty.")
        elif len(summary.split()) < 10:
            quality_penalty += 0.1
            feedback_parts.append("Summary has very few words - may lack substance.")

        scores['quality'] = max(0.0, 1.0 - quality_penalty)

        # 3. Combine scores
        if human_score is not None:
            # Convex combination of oracle and human scores
            base_score = oracle_weight * oracle_score + human_weight * human_score
            scores['human'] = human_score
        else:
            # Oracle only
            base_score = oracle_score

        # Apply quality penalty
        final_score = max(0.0, base_score - quality_penalty)
        scores['final'] = final_score

        # Generate feedback
        if not feedback_parts:
            feedback = "Good RILE preservation and summary quality."
        else:
            feedback = ' '.join(feedback_parts)

        return {
            'score': final_score,
            'feedback': feedback,
            'details': scores,  # Additional detail for debugging
        }

    return metric


def merge(
    oracle_classifier,
    threshold: float = 10.0,
    max_error: float = None,
) -> Callable:
    """
    Create a metric for evaluating merge quality.

    Similar to summarization metric but specifically for merge operations
    where two summaries are combined.

    Args:
        oracle_classifier: Score predictor with predict_rile() method
        threshold: Maximum acceptable RILE drift
        max_error: Maximum error scale for scoring. Defaults to RILE_RANGE (200.0).

    Returns:
        Metric function for merge evaluation
    """
    # Resolve default max_error using lazy import to avoid module-level dependency
    if max_error is None:
        from src.manifesto.constants import RILE_RANGE
        max_error = RILE_RANGE

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        """Evaluate merge quality for RILE preservation.

        Compatible with GEPA's 5-argument signature.
        """
        # Extract inputs
        left_summary = getattr(gold, 'left_summary', '')
        right_summary = getattr(gold, 'right_summary', '')
        ground_truth_rile = getattr(gold, 'ground_truth_rile', 0.0)

        # Extract prediction
        if isinstance(pred, str):
            merged = pred
        else:
            merged = getattr(pred, 'merged_summary', '')
            if not merged:
                merged = getattr(pred, 'summary', '')
            if not merged:
                merged = str(pred)

        feedback_parts = []

        # Check merged preserves RILE
        try:
            merged_rile, confidence, _ = oracle_classifier.predict_rile(merged)
            rile_diff = abs(merged_rile - ground_truth_rile)
            oracle_score = normalize_error_to_score(rile_diff, max_error=max_error)

            if rile_diff > threshold:
                feedback_parts.append(
                    f"Merge lost RILE signal: {merged_rile:.1f} vs expected {ground_truth_rile:.1f}"
                )
        except Exception:
            oracle_score = 0.5

        # Check merge didn't lose too much content
        input_length = len(left_summary) + len(right_summary)
        output_length = len(merged)

        if output_length < input_length * 0.3:
            oracle_score *= 0.9
            feedback_parts.append(
                "Merge compressed significantly - may have lost information."
            )

        feedback = ' '.join(feedback_parts) if feedback_parts else "Good merge quality."

        return {
            'score': oracle_score,
            'feedback': feedback,
        }

    return metric


# =============================================================================
# Simple Metric Factories (DSPy-style)
# =============================================================================

def exact_match(
    gold_field: str = "answer",
    pred_field: str = "answer",
) -> Callable:
    """
    Create a simple exact match metric.

    This follows DSPy's pattern of simple boolean metrics.

    Args:
        gold_field: Field name to get from gold example
        pred_field: Field name to get from prediction

    Returns:
        Metric function returning bool

    Example:
        metric = create_exact_match_metric("label", "label")
        score = metric(gold_example, prediction)  # True/False
    """
    def metric(gold, pred, trace=None) -> bool:
        gold_val = getattr(gold, gold_field, None)
        pred_val = getattr(pred, pred_field, None)
        return gold_val == pred_val

    return metric


def numeric_match(
    gold_field: str,
    pred_field: str,
    tolerance: float = 0.0,
) -> Callable:
    """
    Create a numeric comparison metric with optional tolerance.

    Useful for ordinal or numeric predictions where close-enough is acceptable.

    Args:
        gold_field: Field name for gold value
        pred_field: Field name for predicted value
        tolerance: Maximum allowed difference

    Returns:
        Metric function returning bool

    Example:
        metric = create_numeric_match_metric("score", "score", tolerance=5.0)
        score = metric(gold, pred)  # True if within 5 points
    """
    def metric(gold, pred, trace=None) -> bool:
        try:
            gold_val = float(getattr(gold, gold_field, 0))
            pred_val = float(getattr(pred, pred_field, 0))
            return abs(gold_val - pred_val) <= tolerance
        except (ValueError, TypeError):
            return False

    return metric


def oracle(
    oracle_fn: Callable,
    comparison_fn: Callable = None,
    input_field: str = "output",
    gold_field: str = None,
    default_max_error: float = None,
) -> Callable:
    """
    Create a metric that uses any oracle/classifier as the scoring function.

    This generalizes the oracle-as-metric pattern used in summarization optimization.

    Args:
        oracle_fn: Callable that takes text and returns a score/prediction.
                   Signature: oracle_fn(text) -> Any
        comparison_fn: How to compare oracle output to ground truth.
                       Signature: comparison_fn(oracle_output, gold_value) -> float
                       Default: Uses normalize_error_to_score with default_max_error
        input_field: Field name to extract from prediction for oracle input
        gold_field: Field name for ground truth (if None, uses same as input_field)
        default_max_error: Max error scale for default comparison function.
                          Defaults to RILE_RANGE (200.0) for RILE-style scores.

    Returns:
        Metric function compatible with DSPy optimizers

    Examples:
        # RILE oracle as metric
        from src.manifesto.constants import RILE_RANGE
        metric = create_oracle_metric(
            oracle_fn=rile_oracle.predict,
            comparison_fn=lambda pred, gold: normalize_error_to_score(abs(pred - gold), RILE_RANGE),
            input_field="summary",
            gold_field="ground_truth_rile",
        )

        # Binary oracle (pass/fail)
        metric = create_oracle_metric(
            oracle_fn=law_checker.check,
            comparison_fn=lambda result, _: 1.0 if result.passed else 0.0,
            input_field="summary",
        )
    """
    # Resolve default max_error using lazy import to avoid module-level dependency
    if default_max_error is None:
        from src.manifesto.constants import RILE_RANGE
        default_max_error = RILE_RANGE

    # Default comparison function for numeric scores
    if comparison_fn is None:
        def comparison_fn(oracle_out, gold_val):
            try:
                return normalize_error_to_score(
                    abs(float(oracle_out) - float(gold_val)), default_max_error
                )
            except (ValueError, TypeError):
                return 0.0 if oracle_out != gold_val else 1.0

    gold_field = gold_field or input_field

    def metric(gold, pred, trace=None) -> float:
        # Get input for oracle from prediction
        pred_text = getattr(pred, input_field, None)
        if pred_text is None:
            pred_text = str(pred)

        # Get ground truth from gold example
        gold_value = getattr(gold, gold_field, None)

        try:
            # Run oracle
            oracle_output = oracle_fn(pred_text)

            # Compare oracle output to ground truth
            if gold_value is not None:
                return comparison_fn(oracle_output, gold_value)
            else:
                # If no ground truth, return oracle output directly if it's a score
                if isinstance(oracle_output, (int, float)):
                    return float(oracle_output)
                return 1.0 if oracle_output else 0.0

        except Exception as e:
            return 0.0

    return metric


def combine_metrics(
    metrics: List[Tuple[Callable, float]],
    normalize_weights: bool = True,
) -> Callable:
    """
    Combine multiple metrics with weights.

    Creates a weighted combination of multiple metric functions.

    Args:
        metrics: List of (metric_fn, weight) tuples
        normalize_weights: If True, normalize weights to sum to 1.0

    Returns:
        Combined metric function

    Example:
        combined = combine_metrics([
            (oracle_metric, 0.6),
            (exact_match_metric, 0.3),
            (custom_metric, 0.1),
        ])

        score = combined(gold, pred)  # Weighted average
    """
    if not metrics:
        raise ValueError("At least one metric required")

    # Normalize weights if requested
    if normalize_weights:
        total_weight = sum(w for _, w in metrics)
        if total_weight > 0:
            metrics = [(m, w / total_weight) for m, w in metrics]

    def combined_metric(gold, pred, trace=None) -> float:
        total_score = 0.0

        for metric_fn, weight in metrics:
            try:
                score = metric_fn(gold, pred, trace)
                # Convert bool to float
                if isinstance(score, bool):
                    score = 1.0 if score else 0.0
                # Extract score from dict if needed
                elif isinstance(score, dict):
                    score = score.get('score', 0.0)
                total_score += weight * float(score)
            except Exception:
                # Skip failed metrics
                continue

        return total_score

    return combined_metric


def combine_feedback(
    metrics: List[Tuple[Callable, float, str]],
    normalize_weights: bool = True,
) -> Callable:
    """
    Combine multiple metrics with weights and feedback aggregation.

    Like combine_metrics but returns {'score': float, 'feedback': str}
    for GEPA compatibility.

    Args:
        metrics: List of (metric_fn, weight, name) tuples
        normalize_weights: If True, normalize weights to sum to 1.0

    Returns:
        Combined metric function with feedback

    Example:
        combined = combine_metrics_with_feedback([
            (oracle_metric, 0.6, "oracle"),
            (exact_match, 0.3, "exact"),
            (custom_metric, 0.1, "custom"),
        ])
    """
    if not metrics:
        raise ValueError("At least one metric required")

    # Normalize weights if requested
    if normalize_weights:
        total_weight = sum(w for _, w, _ in metrics)
        if total_weight > 0:
            metrics = [(m, w / total_weight, n) for m, w, n in metrics]

    def combined_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        total_score = 0.0
        feedback_parts = []
        component_scores = {}

        for metric_fn, weight, name in metrics:
            try:
                result = metric_fn(gold, pred, trace)

                # Extract score and feedback
                if isinstance(result, dict):
                    score = result.get('score', 0.0)
                    feedback = result.get('feedback', '')
                elif isinstance(result, bool):
                    score = 1.0 if result else 0.0
                    feedback = ''
                else:
                    score = float(result)
                    feedback = ''

                total_score += weight * score
                component_scores[name] = score

                if feedback:
                    feedback_parts.append(f"{name}: {feedback}")

            except Exception as e:
                component_scores[name] = 0.0
                feedback_parts.append(f"{name}: error ({str(e)[:50]})")

        return {
            'score': total_score,
            'feedback': ' | '.join(feedback_parts) if feedback_parts else 'OK',
            'components': component_scores,
        }

    return combined_metric


# Note: EvaluationResult dataclass has been removed.
# Use continuous score evaluation with oracle_as_metric() or metric() instead.


# =============================================================================
# General Oracle Score Prediction (Scale-Agnostic)
# =============================================================================

def oracle_score_prediction(
    oracle_fn: Callable,
    scale: "ScaleDefinition",
    ground_truth_field: str = 'reference_score',
    summary_field: str = 'summary',
    with_feedback: bool = True,
) -> Callable:
    """
    Create a DSPy metric that compares oracle predictions to ground truth.

    This is a general-purpose metric that works with any bounded scale.
    It uses the scale's `values_to_score()` method to convert prediction
    errors to normalized 0-1 scores.

    Args:
        oracle_fn: Oracle with `predict_score(text)` returning (score, confidence, reasoning).
                   Also supports legacy `predict_rile()` interface for backwards compatibility.
        scale: ScaleDefinition defining the value range for normalization.
               Uses `scale.values_to_score(predicted, ground_truth)` for scoring.
        ground_truth_field: Field name for ground truth on gold example
        summary_field: Field name for summary text on prediction
        with_feedback: Return dict with feedback for GEPA compatibility

    Returns:
        DSPy-compatible metric function (5-arg signature)

    Example:
        from src.ops_engine.training_framework.domains.base import ScaleDefinition

        # Any bounded scale works
        my_scale = ScaleDefinition("sentiment", -1.0, 1.0)
        metric = oracle_score_prediction(
            oracle_fn=sentiment_oracle,
            scale=my_scale,
        )
    """
    # Lazy import to avoid circular dependency
    from src.ops_engine.training_framework.domains.base import ScaleDefinition

    def metric_fn(gold, pred, trace=None, pred_name=None, pred_trace=None):
        # Get ground truth
        ground_truth = getattr(gold, ground_truth_field, None)
        if ground_truth is None:
            ground_truth = getattr(gold, 'label', 0.0)
        try:
            ground_truth = float(ground_truth)
        except (ValueError, TypeError):
            ground_truth = scale.neutral_value or ((scale.min_value + scale.max_value) / 2)

        # Get summary from prediction
        if isinstance(pred, str):
            summary = pred
        else:
            summary = getattr(pred, summary_field, None)
            if summary is None:
                summary = getattr(pred, 'merged_summary', None)
            if summary is None:
                summary = str(pred)

        # Call oracle - support multiple interfaces
        try:
            # Try new general interface first
            if hasattr(oracle_fn, 'predict_score'):
                result = oracle_fn.predict_score(summary)
            # Fall back to legacy RILE interface
            elif hasattr(oracle_fn, 'predict_rile'):
                result = oracle_fn.predict_rile(summary)
            # Direct callable
            else:
                result = oracle_fn(summary)

            # Handle return types
            if isinstance(result, tuple):
                predicted_value = result[0]
                confidence = result[1] if len(result) > 1 else 1.0
                reasoning = result[2] if len(result) > 2 else ""
            else:
                predicted_value = float(result)
                confidence = 1.0
                reasoning = ""

        except Exception as e:
            if with_feedback:
                return {'score': 0.0, 'feedback': f"Oracle error: {str(e)[:50]}"}
            return 0.0

        # Compute score using scale's built-in method
        score = scale.values_to_score(predicted_value, ground_truth)

        if with_feedback:
            error = abs(predicted_value - ground_truth)
            feedback = f"Predicted {predicted_value:.1f}, expected {ground_truth:.1f} (error: {error:.1f})"
            if reasoning:
                feedback += f" - {reasoning}"
            return {'score': score, 'feedback': feedback}

        return score

    return metric_fn


def pairwise_consistency_metric(
    oracle_fn: Callable,
    with_feedback: bool = True,
) -> Callable:
    """
    Create a metric measuring pairwise preference consistency.

    This metric evaluates whether predictions are consistent with oracle
    pairwise comparisons. Useful for GenRM-style preference learning.

    Args:
        oracle_fn: Oracle that can score text. Used to compare two summaries.
        with_feedback: Return dict with feedback for GEPA compatibility

    Returns:
        DSPy-compatible metric function (5-arg signature)

    Example:
        metric = pairwise_consistency_metric(oracle)
        # gold has 'summary_a', 'summary_b', 'preference' (1 or -1)
        score = metric(gold, pred)
    """
    def metric_fn(gold, pred, trace=None, pred_name=None, pred_trace=None):
        # Get the two summaries to compare
        summary_a = getattr(gold, 'summary_a', None)
        summary_b = getattr(gold, 'summary_b', None)
        preference = getattr(gold, 'preference', None)  # 1 if A better, -1 if B better, 0 if equal

        if summary_a is None or summary_b is None:
            if with_feedback:
                return {'score': 0.0, 'feedback': 'Missing summary_a or summary_b'}
            return 0.0

        try:
            # Get oracle scores for both summaries
            if hasattr(oracle_fn, 'predict_score'):
                result_a = oracle_fn.predict_score(summary_a)
                result_b = oracle_fn.predict_score(summary_b)
            elif hasattr(oracle_fn, 'predict_rile'):
                result_a = oracle_fn.predict_rile(summary_a)
                result_b = oracle_fn.predict_rile(summary_b)
            else:
                result_a = oracle_fn(summary_a)
                result_b = oracle_fn(summary_b)

            # Extract scores
            score_a = result_a[0] if isinstance(result_a, tuple) else float(result_a)
            score_b = result_b[0] if isinstance(result_b, tuple) else float(result_b)

            # Determine oracle's preference
            oracle_preference = 1 if score_a > score_b else (-1 if score_b > score_a else 0)

            # Compare with expected preference
            if preference is None:
                # No ground truth preference - return neutral
                score = 0.5
            elif preference == 0 or oracle_preference == 0:
                # Either is neutral - partial credit
                score = 0.75
            elif preference == oracle_preference:
                # Preferences match
                score = 1.0
            else:
                # Preferences conflict
                score = 0.0

            if with_feedback:
                feedback = f"Oracle scores: A={score_a:.1f}, B={score_b:.1f}"
                return {'score': score, 'feedback': feedback}
            return score

        except Exception as e:
            if with_feedback:
                return {'score': 0.0, 'feedback': f"Comparison error: {str(e)[:50]}"}
            return 0.0

    return metric_fn


# =============================================================================
# Domain-Specific Metrics
# =============================================================================
# RILE-specific metrics have been moved to src/manifesto/metrics.py
# Import them directly from there:
#     from src.manifesto.metrics import oracle_rile_prediction

# =============================================================================
# Legacy Aliases
# =============================================================================

# Note: create_classification_metric, evaluate_classifier, and create_violation_metric
# have been removed. Use oracle_as_metric() or metric() for continuous score evaluation.
