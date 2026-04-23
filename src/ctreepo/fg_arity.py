"""Shared f/g arity and token-budget contract.

Canonical rule:
- f consumes one leaf-sized state: ``leaf_size_tokens``.
- g consumes two child states: ``2 * leaf_size_tokens``.
- g may emit their verbatim concatenation: ``2 * leaf_size_tokens``.

This module keeps that rule out of individual backends so DSPy, TRL, teacher
trace generation, and neural/operator backends fail consistently.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FGArityBudget:
    leaf_size_tokens: int
    f_input_tokens: int
    g_input_tokens: int
    g_output_tokens: int


def fg_arity_budget(leaf_size_tokens: int) -> FGArityBudget:
    leaf = int(leaf_size_tokens)
    if leaf <= 0:
        raise ValueError(f"leaf_size_tokens must be positive, got {leaf_size_tokens}")
    return FGArityBudget(
        leaf_size_tokens=leaf,
        f_input_tokens=leaf,
        g_input_tokens=2 * leaf,
        g_output_tokens=2 * leaf,
    )


def auto_g_output_tokens(
    requested: int | None,
    *,
    leaf_size_tokens: int,
) -> int:
    required = fg_arity_budget(int(leaf_size_tokens)).g_output_tokens
    if requested is None or int(requested) <= 0:
        return int(required)
    if int(requested) < int(required):
        raise RuntimeError(
            f"g output budget too small: requested max tokens={requested}, "
            f"but 2 * leaf_size_tokens = {required}. g must be able to emit "
            "a verbatim concatenation of two children."
        )
    return int(requested)


def check_two_child_lm_budget(
    *,
    family_name: str,
    leaf_size_tokens: int,
    lm_context_window_tokens: int,
    max_completion_tokens: int,
    prompt_template_overhead_tokens: int,
) -> None:
    budget = fg_arity_budget(int(leaf_size_tokens))
    if int(max_completion_tokens) < int(budget.g_output_tokens):
        raise RuntimeError(
            f"{family_name}: max_completion_tokens={max_completion_tokens} "
            f"< 2 * leaf_size_tokens = {budget.g_output_tokens}. g must be able "
            "to emit a verbatim concatenation of two children."
        )
    available_input = (
        int(lm_context_window_tokens)
        - int(max_completion_tokens)
        - int(prompt_template_overhead_tokens)
    )
    if int(budget.g_input_tokens) > int(available_input):
        raise RuntimeError(
            f"{family_name}: two-child input budget exceeded. "
            f"2 * leaf_size_tokens = {budget.g_input_tokens}, available input "
            f"budget = {available_input} (= lm_context_window_tokens="
            f"{lm_context_window_tokens} - max_completion_tokens="
            f"{max_completion_tokens} - prompt_template_overhead_tokens="
            f"{prompt_template_overhead_tokens})."
        )
