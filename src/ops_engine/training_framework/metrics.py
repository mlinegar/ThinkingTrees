"""
Evaluation Metrics for Oracle Approximation.

This module provides metrics for evaluating:
- Classification accuracy (standard and distance-weighted)
- Calibration error (confidence vs actual accuracy)
- OPS law compliance rates
- DSPy-compatible metric functions

New API (Preferred):
    from src.ops_engine.scoring import oracle_as_metric, ScoringOracle

    # ScoringOracle -> DSPy metric in one line
    metric = oracle_as_metric(my_scorer)
    optimizer.compile(student, trainset, metric=metric)

Legacy API (for specialized cases):
    from src.ops_engine.training_framework.metrics import create_summarization_metric

    # When you need specialized summarization metrics with quality checks
    metric = create_summarization_metric(oracle_classifier, ...)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable
import math
import warnings

from .core import LabelSpace, Prediction, LawCheckResult

# Re-export score-centric metric converters for convenience
from src.ops_engine.scoring import (
    oracle_as_metric,
    oracle_as_metric_with_feedback,
    ScoringOracle,
    OracleScore,
    normalize_error_to_score,
    BoundedScale,
    PERCENT_SCALE,
)


# =============================================================================
# Generic Metric Factory (Scale-Based)
# =============================================================================

def create_metric(
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


# =============================================================================
# Classification Metrics
# =============================================================================

def classification_accuracy(
    predictions: List[str],
    ground_truth: List[str],
) -> float:
    """
    Standard classification accuracy.

    Args:
        predictions: List of predicted labels
        ground_truth: List of true labels

    Returns:
        Accuracy as fraction [0, 1]
    """
    if not predictions or len(predictions) != len(ground_truth):
        return 0.0

    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(predictions)


def distance_weighted_accuracy(
    predictions: List[str],
    ground_truth: List[str],
    label_space: LabelSpace,
    max_distance: Optional[float] = None,
) -> float:
    """
    Distance-weighted accuracy for ordinal label spaces.

    Gives partial credit based on how close the prediction is to ground truth.
    For categorical spaces, falls back to standard accuracy.

    Args:
        predictions: List of predicted labels
        ground_truth: List of true labels
        label_space: The label space (for distance computation)
        max_distance: Maximum possible distance (auto-computed if None)

    Returns:
        Weighted accuracy as fraction [0, 1]
    """
    if not predictions or len(predictions) != len(ground_truth):
        return 0.0

    if not label_space.is_ordinal:
        # Fall back to standard accuracy for categorical
        return classification_accuracy(predictions, ground_truth)

    # Compute max distance if not provided
    if max_distance is None:
        labels = label_space.labels
        if labels:
            max_distance = label_space.distance(labels[0], labels[-1])
        if max_distance == 0:
            max_distance = 1.0

    # Compute weighted scores
    scores = []
    for pred, true in zip(predictions, ground_truth):
        distance = label_space.distance(pred, true)
        # Score: 1 for exact match, decreasing to 0 at max distance
        score = normalize_error_to_score(distance, max_error=max_distance)
        scores.append(score)

    return sum(scores) / len(scores)


def mean_absolute_error(
    predictions: List[str],
    ground_truth: List[str],
    label_space: LabelSpace,
) -> float:
    """
    Mean absolute error for ordinal predictions.

    Args:
        predictions: List of predicted labels
        ground_truth: List of true labels
        label_space: The label space (for distance computation)

    Returns:
        Mean absolute error
    """
    if not predictions or len(predictions) != len(ground_truth):
        return float('inf')

    errors = [label_space.distance(p, g) for p, g in zip(predictions, ground_truth)]
    return sum(errors) / len(errors)


def within_threshold_accuracy(
    predictions: List[str],
    ground_truth: List[str],
    label_space: LabelSpace,
    threshold: float,
) -> float:
    """
    Fraction of predictions within a threshold distance of ground truth.

    Args:
        predictions: List of predicted labels
        ground_truth: List of true labels
        label_space: The label space
        threshold: Distance threshold

    Returns:
        Fraction within threshold [0, 1]
    """
    if not predictions or len(predictions) != len(ground_truth):
        return 0.0

    within = sum(
        label_space.distance(p, g) <= threshold
        for p, g in zip(predictions, ground_truth)
    )
    return within / len(predictions)


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

def create_classification_metric(
    label_space: LabelSpace,
    weighted: bool = True,
    with_feedback: bool = False,
) -> Callable:
    """
    Create a DSPy-compatible metric function for classification.

    Args:
        label_space: The label space for the task
        weighted: Whether to use distance-weighted accuracy (for ordinal)
        with_feedback: If True, returns {'score': float, 'feedback': str} for GEPA

    Returns:
        Metric function compatible with DSPy optimizers
    """
    # Compute max distance once for ordinal spaces
    max_dist = 1.0
    if label_space.is_ordinal and label_space.labels:
        max_dist = label_space.distance(
            label_space.labels[0],
            label_space.labels[-1]
        )
        if max_dist == 0:
            max_dist = 1.0

    def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        """DSPy metric function with optional GEPA feedback."""
        # Get ground truth from example
        true_label = getattr(example, 'label', None)
        if true_label is None:
            true_label = getattr(example, 'violation_type', None)
        if true_label is None:
            if with_feedback:
                return {'score': 0.0, 'feedback': 'No ground truth label found in example.'}
            return 0.0

        # Get predicted label
        pred_label = getattr(prediction, 'label', None)
        if pred_label is None:
            if with_feedback:
                return {'score': 0.0, 'feedback': 'No prediction label found.'}
            return 0.0

        # Get optional prediction metadata
        pred_confidence = getattr(prediction, 'confidence', 0.5)
        pred_reasoning = getattr(prediction, 'reasoning', '')

        # Compute score
        distance = 0.0
        if weighted and label_space.is_ordinal:
            distance = label_space.distance(pred_label, true_label)
            score = max(0.0, 1.0 - (distance / max_dist))
        else:
            score = 1.0 if pred_label == true_label else 0.0

        if not with_feedback:
            return score

        # Generate diagnostic feedback for GEPA
        feedback_parts = []

        if score < 1.0:
            feedback_parts.append(f"Predicted '{pred_label}' but true label was '{true_label}'.")

            if weighted and label_space.is_ordinal:
                feedback_parts.append(f"RILE distance: {distance:.0f} points.")

                # Check for direction errors (left vs right)
                try:
                    pred_val = float(pred_label)
                    true_val = float(true_label)
                    if (pred_val < 0 and true_val > 0) or (pred_val > 0 and true_val < 0):
                        feedback_parts.append(
                            "Misidentified political direction (left vs right) - "
                            "review ideological indicators more carefully."
                        )
                    elif abs(distance) > 40:
                        feedback_parts.append(
                            "Large error - the text may contain mixed signals or "
                            "domain-specific terminology requiring careful interpretation."
                        )
                except (ValueError, TypeError):
                    pass

            # High confidence but wrong
            if pred_confidence > 0.8 and score < 0.5:
                feedback_parts.append(
                    "High confidence but incorrect - the model is overconfident. "
                    "Consider expressing more uncertainty in ambiguous cases."
                )

            # Low confidence and wrong
            if pred_confidence < 0.3 and score < 0.5:
                feedback_parts.append(
                    "Low confidence and incorrect - more analysis or examples may help."
                )

        feedback = ' '.join(feedback_parts) if feedback_parts else "Correct prediction."

        return {'score': score, 'feedback': feedback}

    return metric


def create_violation_metric() -> Callable:
    """
    Create a DSPy metric for violation detection.

    Scores based on:
    - Correct is_violation prediction
    - Correct violation_type (if is_violation)
    """
    def metric(example, prediction, trace=None, pred_name=None, pred_trace=None) -> float:
        """DSPy metric for violation detection. Compatible with GEPA's 5-argument signature."""
        # Get ground truth
        true_is_violation = getattr(example, 'is_true_violation', None)
        true_type = getattr(example, 'violation_type', 'none')

        # Get prediction
        pred_is_violation = getattr(prediction, 'is_violation', None)
        pred_type = getattr(prediction, 'violation_type', 'none')

        if true_is_violation is None or pred_is_violation is None:
            return 0.0

        # Score components
        score = 0.0

        # Is violation correct? (50% weight)
        if pred_is_violation == true_is_violation:
            score += 0.5

        # Violation type correct? (50% weight, only if is_violation)
        if true_is_violation:
            if str(pred_type).lower() == str(true_type).lower():
                score += 0.5
        else:
            # For non-violations, full credit if we correctly said "no violation"
            if not pred_is_violation:
                score += 0.5

        return score

    return metric


def create_law_compliance_metric(
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
# Advanced DSPy Metrics (Trace-based, LLM Judge, Composite)
# =============================================================================

def create_classification_metric_with_trace(
    label_space: LabelSpace,
    weighted: bool = True,
    with_feedback: bool = True,
) -> Callable:
    """
    Create a metric that uses the DSPy trace for deeper validation.

    The trace parameter provides access to intermediate DSPy module calls,
    allowing validation of reasoning quality, not just final answers.

    This follows DSPy metrics best practices from dspy.ai/learn/evaluation/metrics/

    Args:
        label_space: The label space for the task
        weighted: Whether to use distance-weighted accuracy (for ordinal)
        with_feedback: If True, returns {'score': float, 'feedback': str} for GEPA

    Returns:
        Metric function compatible with DSPy optimizers (including GEPA)
    """
    # Compute max distance once for ordinal spaces
    max_dist = 1.0
    if label_space.is_ordinal and label_space.labels:
        max_dist = label_space.distance(
            label_space.labels[0],
            label_space.labels[-1]
        )
        if max_dist == 0:
            max_dist = 1.0

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        """
        DSPy metric with trace-based reasoning validation.

        Compatible with GEPA's 5-argument signature.

        Args:
            gold: Ground truth example
            pred: Prediction
            trace: Optional list of (module, inputs, outputs) tuples from DSPy
            pred_name: Optional predictor name (for GEPA)
            pred_trace: Optional predictor trace (for GEPA)

        Returns:
            Score (float) or {'score': float, 'feedback': str} if with_feedback
        """
        # Get ground truth from example
        true_label = getattr(gold, 'label', None)
        if true_label is None:
            true_label = getattr(gold, 'violation_type', None)
        if true_label is None:
            if with_feedback:
                return {'score': 0.0, 'feedback': 'No ground truth label found.'}
            return 0.0

        # Get predicted label
        pred_label = getattr(pred, 'label', None)
        if pred_label is None:
            if with_feedback:
                return {'score': 0.0, 'feedback': 'No prediction label found.'}
            return 0.0

        # Compute base score
        distance = 0.0
        if weighted and label_space.is_ordinal:
            distance = label_space.distance(pred_label, true_label)
            base_score = max(0.0, 1.0 - (distance / max_dist))
        else:
            base_score = 1.0 if pred_label == true_label else 0.0

        # Use trace for reasoning validation (when available during optimization)
        reasoning_penalty = 0.0
        feedback_parts = []

        if trace is not None:
            # Trace contains intermediate module calls
            for module_call in trace:
                # Handle different trace formats
                if isinstance(module_call, tuple) and len(module_call) >= 3:
                    module_name, inputs, outputs = module_call[:3]
                elif hasattr(module_call, 'outputs'):
                    outputs = module_call.outputs
                else:
                    continue

                # Check for reasoning quality issues
                reasoning = getattr(outputs, 'reasoning', '')
                if not reasoning and isinstance(outputs, dict):
                    reasoning = outputs.get('reasoning', '')

                if reasoning:
                    # Too brief
                    if len(reasoning) < 50:
                        reasoning_penalty += 0.05
                        feedback_parts.append("Reasoning too brief - elaborate on evidence.")

                    # Missing causal connectors
                    reasoning_lower = reasoning.lower()
                    has_causal = any(word in reasoning_lower for word in
                                    ['therefore', 'because', 'since', 'thus', 'hence'])
                    if not has_causal:
                        reasoning_penalty += 0.02
                        feedback_parts.append("Missing causal connectors in reasoning.")

                    # Check for political direction consistency (RILE-specific)
                    if label_space.is_ordinal:
                        try:
                            pred_val = float(pred_label)
                            pred_direction = 'left' if pred_val < 0 else 'right'

                            mentions_left = any(w in reasoning_lower for w in
                                              ['left', 'progressive', 'socialist', 'liberal'])
                            mentions_right = any(w in reasoning_lower for w in
                                               ['right', 'conservative', 'traditional', 'market'])

                            # Inconsistency check
                            if pred_direction == 'left' and mentions_right and not mentions_left:
                                reasoning_penalty += 0.1
                                feedback_parts.append(
                                    "Reasoning mentions right-wing concepts but predicts left - inconsistent."
                                )
                            elif pred_direction == 'right' and mentions_left and not mentions_right:
                                reasoning_penalty += 0.1
                                feedback_parts.append(
                                    "Reasoning mentions left-wing concepts but predicts right - inconsistent."
                                )
                        except (ValueError, TypeError):
                            pass

        final_score = max(0.0, base_score - reasoning_penalty)

        if not with_feedback:
            return final_score

        # Add base error feedback
        if base_score < 1.0:
            feedback_parts.insert(0, f"Predicted '{pred_label}' but true was '{true_label}'.")
            if weighted and label_space.is_ordinal:
                feedback_parts.insert(1, f"Distance: {distance:.0f} points.")

        feedback = ' '.join(feedback_parts) if feedback_parts else "Correct with good reasoning."

        return {'score': final_score, 'feedback': feedback}

    return metric


def create_llm_judge_metric(judge_lm=None) -> Callable:
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


def create_composite_metric(
    label_space: LabelSpace,
    use_trace: bool = True,
    use_llm_judge: bool = False,
    judge_weight: float = 0.3,
    judge_lm=None,
) -> Callable:
    """
    Create a composite metric combining multiple evaluation strategies.

    Combines:
    - Distance-based scoring (always used)
    - Trace-based reasoning validation (optional)
    - LLM judge for nuanced evaluation (optional, expensive)

    Args:
        label_space: Label space for distance calculations
        use_trace: Whether to validate reasoning via trace
        use_llm_judge: Whether to use LLM as judge (expensive but nuanced)
        judge_weight: Weight for LLM judge score when used (0.0-1.0)
        judge_lm: Optional LM for judge (uses default if None)

    Returns:
        Composite metric function
    """
    # Create component metrics
    distance_metric = create_classification_metric_with_trace(
        label_space, weighted=True, with_feedback=True
    ) if use_trace else create_classification_metric(
        label_space, weighted=True, with_feedback=True
    )

    llm_judge = create_llm_judge_metric(judge_lm) if use_llm_judge else None

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        """
        Composite metric combining multiple evaluation approaches.

        Compatible with GEPA's 5-argument signature.

        Args:
            gold: Ground truth example
            pred: Prediction
            trace: Optional DSPy trace
            pred_name: Optional predictor name (for GEPA)
            pred_trace: Optional predictor trace (for GEPA)

        Returns:
            {'score': float, 'feedback': str}
        """
        # Always compute distance-based score
        dist_result = distance_metric(gold, pred, trace=trace if use_trace else None)

        if isinstance(dist_result, dict):
            dist_score = dist_result.get('score', 0.0)
            feedback_parts = [dist_result.get('feedback', '')]
        else:
            dist_score = dist_result
            feedback_parts = []

        # Optionally add LLM judge
        if llm_judge and use_llm_judge:
            try:
                judge_result = llm_judge(gold, pred, trace=trace)
                judge_score = judge_result.get('score', 0.5)
                judge_feedback = judge_result.get('feedback', '')

                if judge_feedback:
                    feedback_parts.append(f"Judge: {judge_feedback}")

                # Weighted combination
                final_score = (1 - judge_weight) * dist_score + judge_weight * judge_score
            except Exception:
                # Fall back to distance-only on judge failure
                final_score = dist_score
        else:
            final_score = dist_score

        return {
            'score': final_score,
            'feedback': ' | '.join(filter(None, feedback_parts)),
        }

    return metric


# =============================================================================
# Summarization Metrics (for Two-Step Iterative Optimization)
# =============================================================================

def create_summarization_metric(
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


def create_merge_metric(
    oracle_classifier,
    threshold: float = 10.0,
    max_error: float = RILE_RANGE,
) -> Callable:
    """
    Create a metric for evaluating merge quality.

    Similar to summarization metric but specifically for merge operations
    where two summaries are combined.

    Args:
        oracle_classifier: Trained RILEOracleClassifier
        threshold: Maximum acceptable RILE drift

    Returns:
        Metric function for merge evaluation
    """

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

def create_exact_match_metric(
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


def create_numeric_match_metric(
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


def create_oracle_metric(
    oracle_fn: Callable,
    comparison_fn: Callable = None,
    input_field: str = "output",
    gold_field: str = None,
    default_max_error: float = RILE_RANGE,
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
                          Default is RILE_RANGE (200.0) for RILE-style scores.

    Returns:
        Metric function compatible with DSPy optimizers

    Examples:
        # RILE oracle as metric
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


def combine_metrics_with_feedback(
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


# =============================================================================
# Aggregate Metrics
# =============================================================================

@dataclass
class EvaluationResult:
    """Comprehensive evaluation results."""
    accuracy: float
    weighted_accuracy: float
    mae: float
    calibration_ece: float
    calibration_mce: float
    law_compliance: Dict[str, float]
    overall_compliance: float

    # Threshold accuracies (for ordinal)
    within_5: Optional[float] = None
    within_10: Optional[float] = None
    within_20: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'accuracy': self.accuracy,
            'weighted_accuracy': self.weighted_accuracy,
            'mae': self.mae,
            'calibration_ece': self.calibration_ece,
            'calibration_mce': self.calibration_mce,
            'law_compliance': self.law_compliance,
            'overall_compliance': self.overall_compliance,
            'within_5': self.within_5,
            'within_10': self.within_10,
            'within_20': self.within_20,
        }


def evaluate_classifier(
    predictions: List[Prediction],
    ground_truth: List[str],
    label_space: LabelSpace,
    law_checks: Optional[List[LawCheckResult]] = None,
) -> EvaluationResult:
    """
    Comprehensive evaluation of classifier performance.

    Args:
        predictions: List of Prediction objects
        ground_truth: List of true labels
        label_space: The label space
        law_checks: Optional law check results

    Returns:
        EvaluationResult with all metrics
    """
    pred_labels = [p.label for p in predictions]

    # Basic metrics
    acc = classification_accuracy(pred_labels, ground_truth)
    weighted_acc = distance_weighted_accuracy(pred_labels, ground_truth, label_space)
    mae = mean_absolute_error(pred_labels, ground_truth, label_space)

    # Calibration
    cal = calibration_error(predictions, ground_truth)

    # Law compliance
    if law_checks:
        law_comp = law_compliance_rate(law_checks)
        overall_comp = overall_compliance_rate(law_checks)
    else:
        law_comp = {}
        overall_comp = 1.0

    # Threshold accuracies (for ordinal)
    within_5 = within_10 = within_20 = None
    if label_space.is_ordinal:
        within_5 = within_threshold_accuracy(pred_labels, ground_truth, label_space, 5.0)
        within_10 = within_threshold_accuracy(pred_labels, ground_truth, label_space, 10.0)
        within_20 = within_threshold_accuracy(pred_labels, ground_truth, label_space, 20.0)

    return EvaluationResult(
        accuracy=acc,
        weighted_accuracy=weighted_acc,
        mae=mae,
        calibration_ece=cal['ece'],
        calibration_mce=cal['mce'],
        law_compliance=law_comp,
        overall_compliance=overall_comp,
        within_5=within_5,
        within_10=within_10,
        within_20=within_20,
    )
