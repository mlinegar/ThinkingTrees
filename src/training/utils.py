"""
Shared utilities for training modules.

This module provides common functions used across training scripts.
"""


def normalize_error_to_01(error: float, scale_range: float) -> float:
    """
    Normalize an error value to the [0, 1] range.

    Args:
        error: Raw error value (e.g., absolute difference between scores)
        scale_range: The range of the scale (e.g., 200 for RILE scale from -100 to +100)

    Returns:
        Normalized error in [0, 1] range

    Example:
        >>> normalize_error_to_01(10.0, 200.0)  # 10-point error on 200-point scale
        0.05
    """
    return min(1.0, max(0.0, error / scale_range))


def compute_mae(predictions: list, ground_truths: list) -> float:
    """
    Compute Mean Absolute Error between predictions and ground truths.

    Args:
        predictions: List of predicted values
        ground_truths: List of ground truth values

    Returns:
        Mean absolute error
    """
    if not predictions or not ground_truths:
        return 0.0
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have same length")

    total_error = sum(abs(p - g) for p, g in zip(predictions, ground_truths))
    return total_error / len(predictions)
