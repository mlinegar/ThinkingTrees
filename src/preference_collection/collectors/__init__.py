"""
Concrete PreferenceCollector implementations.

Importing this package triggers registration of all built-in collectors.
"""

from src.preference_collection.collectors.oracle import OracleCollector  # noqa: F401

# Lazy imports for optional-dependency collectors
def _register_optional():
    """Import collectors with optional dependencies without failing."""
    try:
        from src.preference_collection.collectors.llm_judge import LLMJudgeCollector  # noqa: F401
    except ImportError:
        pass
    try:
        from src.preference_collection.collectors.composite import CompositeCollector  # noqa: F401
    except ImportError:
        pass
    try:
        from src.preference_collection.collectors.human import HumanCollector  # noqa: F401
    except ImportError:
        pass

_register_optional()
