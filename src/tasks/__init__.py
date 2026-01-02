"""
Task registry.

Tasks describe what we do with documents (summarization, scoring, extraction).
"""

from src.tasks.base import (
    TaskPlugin,
    AbstractTask,
)
from src.tasks.registry import (
    TaskRegistry,
    register_task,
)

# Helper functions using registry
def get_task(name: str):
    """Get a task by name from the registry."""
    return TaskRegistry.get(name)

def list_tasks():
    """List all registered tasks."""
    return TaskRegistry.list_tasks()

from .prompting import PromptBuilders, default_merge_prompt, default_summarize_prompt, parse_numeric_score


__all__ = [
    "TaskPlugin",
    "AbstractTask",
    "TaskRegistry",
    "register_task",
    "get_task",
    "list_tasks",
    "PromptBuilders",
    "default_merge_prompt",
    "default_summarize_prompt",
    "parse_numeric_score",
]
