"""
Task plugins submodule for the training framework.

This module provides a plugin system for task-specific training integration.
Each task represents a different use case (what you're predicting) that can
plug into the OPS training framework.

Key concept: Tasks represent the end goal, not the method.
- manifesto_rile → predicts political positioning (-100 to +100)
- document_analysis → predicts content preservation quality (0 to 1)

The pipeline (tree building, summarization) is the METHOD that's constant.

Supported Tasks:
- manifesto_rile: Political manifesto RILE scoring
- document_analysis: Generic content preservation evaluation (default)

Usage:
    from ops_engine.training_framework.tasks import (
        TaskRegistry,
        get_task,
        list_tasks,
    )

    # Get a task instance
    task = get_task("document_analysis")

    # Create task-specific components
    metric = task.create_metric(with_feedback=True)
    predictor = task.create_predictor()
    training_source = task.create_training_source(results)

    # List available tasks
    available = list_tasks()
"""

from typing import Any, Dict, Optional

# Import base classes and abstractions
from .base import (
    TaskPlugin,
    AbstractTask,
    OutputType,
    ScaleDefinition,
    LabelDefinition,
    TaskConfig,
    UnifiedTrainingSource,
    UnifiedResult,
    DEFAULT_UNIFIED_RUBRIC,
)

# Import registry
from .registry import (
    TaskRegistry,
    register_task,
)

# Import task implementations (registration happens on import)
from src.tasks.manifesto.task import ManifestoTask

# Default task is document_analysis (general purpose)
from src.config.settings import DEFAULT_TASK
from .document_analysis import (
    DocumentAnalysisTask,
    DocumentAnalysisTrainingSource,
    PRESERVATION_SCALE,
    PreservationScorer,
    OPS_PRESERVATION_RUBRIC,
)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_task(name: str, **kwargs) -> TaskPlugin:
    """
    Get a task instance by name.

    Args:
        name: Task name (e.g., 'manifesto_rile')
        **kwargs: Arguments to pass to task constructor

    Returns:
        Task instance

    Raises:
        KeyError: If task not found
    """
    return TaskRegistry.get(name, **kwargs)


def list_tasks() -> Dict[str, Dict[str, Any]]:
    """
    List all registered tasks.

    Returns:
        Dict mapping task names to metadata
    """
    return TaskRegistry.list_tasks()


def is_task_registered(name: str) -> bool:
    """
    Check if a task is registered.

    Args:
        name: Task name

    Returns:
        True if registered, False otherwise
    """
    return TaskRegistry.is_registered(name)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Base classes and abstractions
    'TaskPlugin',
    'AbstractTask',
    'OutputType',
    'ScaleDefinition',
    'LabelDefinition',
    'TaskConfig',
    'UnifiedTrainingSource',
    'UnifiedResult',
    'DEFAULT_UNIFIED_RUBRIC',

    # Registry
    'TaskRegistry',
    'register_task',

    # Task implementations
    'ManifestoTask',
    'DocumentAnalysisTask',
    'DocumentAnalysisTrainingSource',
    'PreservationScorer',
    'DEFAULT_TASK',
    'PRESERVATION_SCALE',
    'OPS_PRESERVATION_RUBRIC',

    # Convenience functions
    'get_task',
    'list_tasks',
    'is_task_registered',
]
