"""
Task registry.

Tasks describe what we do with documents (summarization, scoring, extraction).
"""

from src.ops_engine.training_framework.domains import (
    DomainPlugin as TaskPlugin,
    AbstractDomain as AbstractTask,
    DomainRegistry as TaskRegistry,
    get_domain as get_task,
    list_domains as list_tasks,
)

from .prompting import PromptBuilders, default_merge_prompt, default_summarize_prompt, parse_numeric_score


__all__ = [
    "TaskPlugin",
    "AbstractTask",
    "TaskRegistry",
    "get_task",
    "list_tasks",
    "PromptBuilders",
    "default_merge_prompt",
    "default_summarize_prompt",
    "parse_numeric_score",
]
