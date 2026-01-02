"""
Task registry for managing task plugins.

This module provides a central registry for task implementations,
allowing dynamic discovery and selection of tasks.
"""

import logging
from typing import Dict, List, Optional, Type, Union, Iterable

from .base import TaskPlugin, AbstractTask

logger = logging.getLogger(__name__)


class TaskRegistry:
    """
    Registry for task implementations.

    Provides a central place to register and retrieve task plugins.
    Tasks can be registered by name and retrieved dynamically.
    """

    _tasks: Dict[str, Type[TaskPlugin]] = {}
    _instances: Dict[str, TaskPlugin] = {}

    @classmethod
    def register(cls, name: str, task_class: Type[TaskPlugin]) -> None:
        """
        Register a task implementation.

        Args:
            name: Unique name for the task
            task_class: Task class implementing TaskPlugin
        """
        if name in cls._tasks:
            logger.warning(f"Overwriting existing task: {name}")
        cls._tasks[name] = task_class
        logger.debug(f"Registered task: {name}")

    @classmethod
    def get_class(cls, name: str) -> Type[TaskPlugin]:
        """
        Get task class by name.

        Args:
            name: Task name

        Returns:
            Task class

        Raises:
            KeyError: If task not found
        """
        if name not in cls._tasks:
            available = list(cls._tasks.keys())
            raise KeyError(f"Unknown task: {name}. Available: {available}")
        return cls._tasks[name]

    @classmethod
    def get(cls, name: str, **kwargs) -> TaskPlugin:
        """
        Get or create a task instance.

        Args:
            name: Task name
            **kwargs: Arguments to pass to task constructor

        Returns:
            Task instance
        """
        # Create new instance with provided kwargs
        task_class = cls.get_class(name)
        return task_class(**kwargs)

    @classmethod
    def get_singleton(cls, name: str, **kwargs) -> TaskPlugin:
        """
        Get a singleton task instance (cached).

        Args:
            name: Task name
            **kwargs: Arguments to pass to task constructor (only used on first call)

        Returns:
            Cached task instance
        """
        if name not in cls._instances:
            cls._instances[name] = cls.get(name, **kwargs)
        return cls._instances[name]

    @classmethod
    def list_tasks(cls) -> Dict[str, Dict]:
        """
        List all registered tasks with their metadata.

        Returns:
            Dict mapping names to task metadata
        """
        result = {}
        for name, task_class in cls._tasks.items():
            try:
                # Try to get info from class
                result[name] = {
                    'class': task_class.__name__,
                    'module': task_class.__module__,
                }
            except Exception:
                result[name] = {
                    'class': str(task_class),
                    'module': 'unknown',
                }
        return result

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a task is registered."""
        return name in cls._tasks

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tasks (mainly for testing)."""
        cls._tasks.clear()
        cls._instances.clear()


def register_task(name: Union[str, Iterable[str]]):
    """
    Decorator to register a task class.

    Usage:
        @register_task("my_task")
        class MyTask(AbstractTask):
            ...
    """
    def decorator(cls: Type[TaskPlugin]) -> Type[TaskPlugin]:
        names = [name] if isinstance(name, str) else list(name)
        for item in names:
            TaskRegistry.register(item, cls)
        return cls
    return decorator
