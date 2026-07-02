"""
Generalized preference collection for ThinkingTrees.

Type-agnostic preference system supporting pairwise preferences, scalar ratings,
written critiques, and arbitrary combinations. Works with LLM judges,
oracle scoring functions, and human reviewers (via API).

Core Types:
    PreferenceRequest     -- declares what preference data is wanted
    PreferenceResponse    -- carries collected preference data
    PreferenceDimension   -- specifies a single preference dimension

Protocol:
    PreferenceCollector   -- generalized collection interface

Built-in Collectors:
    PreferenceDeriverAdapter -- wraps existing PreferenceDeriver
    OracleCollector          -- oracle scoring function
    LLMJudgeCollector        -- LLM-based multi-dimensional preference data
    HumanCollector           -- queues for human review via API
    CompositeCollector       -- combines multiple collectors

Registry:
    register_collector(name)  -- decorator to register collectors
    get_collector(name, ...)  -- factory to instantiate by name
    list_collectors()         -- list registered names

Usage:
    from src.preference_collection import (
        PreferenceRequest,
        PreferenceResponse,
        get_collector,
    )

    collector = get_collector("oracle", oracle_predict=my_fn)
    request = PreferenceRequest(
        request_id="r1",
        text_a="Summary to rate...",
        original_text="Source text...",
        rubric="Preserve key arguments",
    )
    response = collector.collect(request)
    print(response.to_dspy_metric())
"""

from src.preference_collection.types import (
    PreferenceDimension,
    PreferenceRequest,
    PreferenceResponse,
    preference_dataset_from_responses,
)
from src.preference_collection.collector import (
    PreferenceCollector,
    PreferenceDeriverAdapter,
    get_collector,
    list_collectors,
    register_collector,
)
from src.preference_collection.store import PreferenceStore
from src.preference_collection.collectors.human import HumanCollector

# Import concrete collectors to trigger registration
import src.preference_collection.collectors  # noqa: F401

__all__ = [
    "PreferenceDimension",
    "PreferenceRequest",
    "PreferenceResponse",
    "PreferenceCollector",
    "PreferenceStore",
    "HumanCollector",
    "PreferenceDeriverAdapter",
    "preference_dataset_from_responses",
    "get_collector",
    "list_collectors",
    "register_collector",
]
