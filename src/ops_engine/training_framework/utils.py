"""
Utility functions for the training framework.

This module provides helper functions for:
- Example selection (anti-anchoring strategies)
- Data processing utilities
"""

import random
from typing import List, Any, Optional, Union, Callable


def select_median_example(
    examples: List[Any],
    score_key: Union[str, Callable[[Any], float]] = 'score'
) -> Optional[Any]:
    """
    Select an example near the median score to use as the first example.

    This prevents LLM anchoring on extreme values. When LLMs see specific
    values as examples (e.g., "-35.0"), they tend to disproportionately
    output those values. By selecting from the middle of the distribution,
    we avoid this bias.

    Adds slight randomization to avoid always picking the exact same example.

    Args:
        examples: List of examples with scores
        score_key: Either a string key/attribute name for the score value,
                   or a callable that extracts the score from an example

    Returns:
        An example from the middle third of the score distribution,
        or None if examples is empty

    Examples:
        # With dict examples
        examples = [{'text': 'a', 'score': -50}, {'text': 'b', 'score': 0}, ...]
        median_ex = select_median_example(examples, score_key='score')

        # With object examples
        median_ex = select_median_example(examples, score_key='rile_score')

        # With custom extractor
        median_ex = select_median_example(examples, score_key=lambda x: x.ground_truth)
    """
    if not examples:
        return None

    if len(examples) == 1:
        return examples[0]

    # Define score extractor
    if callable(score_key):
        get_score = score_key
    else:
        def get_score(ex: Any) -> float:
            if isinstance(ex, dict):
                return float(ex.get(score_key, 0))
            return float(getattr(ex, score_key, 0))

    # Sort by score
    try:
        sorted_examples = sorted(examples, key=get_score)
    except (TypeError, ValueError):
        # If sorting fails, return random example
        return random.choice(examples)

    # Pick from middle third with slight randomization
    n = len(sorted_examples)
    middle_start = n // 3
    middle_end = 2 * n // 3

    # Ensure we have at least one element in the middle range
    if middle_start >= middle_end:
        return sorted_examples[n // 2]

    return random.choice(sorted_examples[middle_start:middle_end])


def order_examples_median_first(
    examples: List[Any],
    score_key: Union[str, Callable[[Any], float]] = 'score'
) -> List[Any]:
    """
    Reorder examples so that a median-score example comes first.

    This is useful when providing few-shot examples to an LLM - placing
    a median example first reduces anchoring on extreme values.

    Args:
        examples: List of examples with scores
        score_key: Key/attribute name or callable for extracting scores

    Returns:
        New list with a median example first, followed by remaining
        examples in their original order

    Example:
        # Original: [ex_low, ex_high, ex_mid]
        # Returns:  [ex_mid, ex_low, ex_high]
        ordered = order_examples_median_first(examples, score_key='score')
    """
    if not examples or len(examples) <= 1:
        return list(examples)

    median_ex = select_median_example(examples, score_key)
    if median_ex is None:
        return list(examples)

    # Build new list with median first
    result = [median_ex]
    for ex in examples:
        if ex is not median_ex:
            result.append(ex)

    return result
