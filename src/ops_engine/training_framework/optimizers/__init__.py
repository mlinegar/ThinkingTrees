"""
Optimizers submodule for the training framework.

This module provides a registry of optimizer implementations that can be
dynamically selected based on dataset size, computational budget, or
user preference.

Supported Optimizers:
- gepa: GEPA optimizer (default) - sophisticated prompt optimization
- bootstrap: BootstrapFewShot - basic few-shot optimization
- bootstrap_random_search: BootstrapFewShotWithRandomSearch - parallel search
- labeled_fewshot: LabeledFewShot - direct labeled examples
- mipro: MIPROv2 - multi-stage instruction optimization

Usage:
    from ops_engine.training_framework.optimizers import (
        OptimizerRegistry,
        get_optimizer,
        auto_select_optimizer,
    )

    # Get optimizer by name
    optimizer = get_optimizer("bootstrap_random_search", config)
    compiled = optimizer.compile(student, trainset, metric=metric)

    # Auto-select based on dataset size
    optimizer_name = auto_select_optimizer(len(trainset), config)
    optimizer = get_optimizer(optimizer_name, config)

Note: The legacy OracleOptimizer class is still available in the parent module
for backward compatibility:
    from ops_engine.training_framework import OracleOptimizer
"""

from typing import Optional, TYPE_CHECKING

# Import base classes
from .base import (
    BaseOptimizer,
    AbstractOptimizer,
    OptimizationResult,
)

# Import registry
from .registry import (
    OptimizerRegistry,
    register_optimizer,
)

# Import optimizer implementations (registration happens on import)
from .gepa import GEPAOptimizer
from .bootstrap import (
    BootstrapOptimizer,
    BootstrapRandomSearchOptimizer,
    LabeledFewShotOptimizer,
)
from .mipro import MIPROOptimizer
from .parallel import (
    ParallelModuleOptimizer,
    ModuleOptimizationConfig,
    ParallelOptimizationResult,
    create_parallel_optimizer,
)

if TYPE_CHECKING:
    from ..config import OptimizationConfig


# =============================================================================
# Convenience Functions
# =============================================================================

def get_optimizer(
    name: str,
    config: Optional['OptimizationConfig'] = None,
) -> BaseOptimizer:
    """
    Get an optimizer instance by name.

    Args:
        name: Optimizer name (e.g., 'gepa', 'bootstrap_random_search')
        config: Optional configuration

    Returns:
        Optimizer instance

    Raises:
        KeyError: If optimizer not found
    """
    return OptimizerRegistry.get(name, config)


def auto_select_optimizer(
    dataset_size: int,
    config: Optional['OptimizationConfig'] = None,
) -> str:
    """
    Auto-select the best optimizer based on dataset size.

    Selection follows DSPy best practices:
    - dataset_size <= 10: bootstrap
    - dataset_size <= 50: bootstrap_random_search
    - dataset_size <= 200: mipro
    - dataset_size > 200: gepa

    Args:
        dataset_size: Number of training examples
        config: Optional config with custom thresholds

    Returns:
        Name of recommended optimizer
    """
    return OptimizerRegistry.auto_select(dataset_size, config)


def list_optimizers() -> dict:
    """
    List all registered optimizers.

    Returns:
        Dict mapping optimizer names to metadata
    """
    return OptimizerRegistry.list_optimizers()


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Base classes (for extending)
    'BaseOptimizer',
    'AbstractOptimizer',
    'OptimizationResult',

    # Registry (primary API)
    'OptimizerRegistry',
    'register_optimizer',

    # Convenience functions (primary API)
    'get_optimizer',
    'auto_select_optimizer',
    'list_optimizers',

    # Parallel optimization
    'ParallelModuleOptimizer',
    'ModuleOptimizationConfig',
    'ParallelOptimizationResult',
    'create_parallel_optimizer',
]

# Note: Individual optimizer classes (GEPAOptimizer, BootstrapOptimizer, etc.)
# are importable but not in __all__. Use get_optimizer("name") instead.
