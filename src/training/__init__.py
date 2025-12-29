"""
Training pipeline module for OPS optimization.

This module provides CLI entry points for running training pipelines,
integrating the OPS training framework with task + dataset plugins.
"""


def __getattr__(name):
    """Lazy import to avoid circular import issues when running with -m flag."""
    if name == 'run_training_pipeline':
        from .run_pipeline import run_training_pipeline
        return run_training_pipeline
    elif name == 'main':
        from .run_pipeline import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ['run_training_pipeline', 'main']
