"""
Test configuration loading utility.

Loads test settings from YAML configuration files for consistent test behavior.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


_CONFIG_CACHE: Dict[str, Any] = {}


def get_test_config_path() -> Path:
    """Get path to test configuration file."""
    return Path(__file__).parent / "config" / "test_settings.yaml"


def load_test_config(force_reload: bool = False) -> Dict[str, Any]:
    """
    Load test configuration from YAML file.

    Args:
        force_reload: Force reloading from disk

    Returns:
        Configuration dictionary
    """
    global _CONFIG_CACHE

    if not force_reload and _CONFIG_CACHE:
        return _CONFIG_CACHE

    config_path = get_test_config_path()
    if config_path.exists():
        with open(config_path) as f:
            _CONFIG_CACHE = yaml.safe_load(f) or {}
    else:
        _CONFIG_CACHE = {}

    return _CONFIG_CACHE


def get_chunking_config(profile: str = "default") -> Dict[str, Any]:
    """
    Get chunking configuration for a specific profile.

    Args:
        profile: Configuration profile name (default, small, with_overlap)

    Returns:
        Chunking configuration dictionary
    """
    config = load_test_config()
    return config.get("chunking", {}).get(profile, {})


def get_builder_config(profile: str = "default") -> Dict[str, Any]:
    """
    Get builder configuration for a specific profile.

    Args:
        profile: Configuration profile name

    Returns:
        Builder configuration dictionary
    """
    config = load_test_config()
    return config.get("builder", {}).get(profile, {})


def get_auditor_config(profile: str = "default") -> Dict[str, Any]:
    """
    Get auditor configuration for a specific profile.

    Args:
        profile: Configuration profile name

    Returns:
        Auditor configuration dictionary
    """
    config = load_test_config()
    return config.get("auditor", {}).get(profile, {})


def get_sample_text(name: str = "short") -> str:
    """
    Get sample text by name.

    Args:
        name: Text name (short, medium, long)

    Returns:
        Sample text string
    """
    config = load_test_config()
    texts = config.get("sample_texts", {})
    return texts.get(name, "")


def get_rubric(name: str = "default") -> str:
    """
    Get sample rubric by name.

    Args:
        name: Rubric name

    Returns:
        Rubric string
    """
    config = load_test_config()
    rubrics = config.get("rubrics", {})
    return rubrics.get(name, "")


# Convenience class for accessing config in tests
class ConfigAccessor:
    """
    Convenience class for accessing test configuration.

    Example:
        cfg = ConfigAccessor()
        chunker = DocumentChunker(**cfg.chunking("small"))
    """

    def __init__(self):
        self._config = load_test_config()

    def chunking(self, profile: str = "default") -> Dict[str, Any]:
        """Get chunking config with parameter name mapping."""
        raw = get_chunking_config(profile)
        return {
            "max_chunk_chars": raw.get("max_chars", 500),
            "min_chunk_chars": raw.get("min_chars", 50),
            "overlap_chars": raw.get("overlap", 0),
        }

    def builder(self, profile: str = "default") -> Dict[str, Any]:
        """Get builder config."""
        return get_builder_config(profile)

    def auditor(self, profile: str = "default") -> Dict[str, Any]:
        """Get auditor config."""
        return get_auditor_config(profile)

    def text(self, name: str = "short") -> str:
        """Get sample text."""
        return get_sample_text(name)

    def rubric(self, name: str = "default") -> str:
        """Get sample rubric."""
        return get_rubric(name)
