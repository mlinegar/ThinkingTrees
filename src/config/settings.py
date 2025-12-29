"""
Settings loader for YAML configuration.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# Default server URLs
DEFAULT_TASK_MODEL_URL = "http://localhost:8000/v1"
DEFAULT_GENRM_URL = "http://localhost:8001/v1"


def default_settings_path() -> Path:
    """Return default settings.yaml path within the repo."""
    return Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def load_settings(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load settings from a YAML file.

    Args:
        path: Optional path override. Defaults to repo config/settings.yaml.

    Returns:
        Parsed settings dict (empty if file missing).
    """
    settings_path = Path(path) if path else default_settings_path()
    if not settings_path.exists():
        return {}

    with open(settings_path, "r") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def get_task_model_url(settings: Optional[Dict[str, Any]] = None) -> str:
    """
    Get the task model (vLLM) server URL.

    Priority:
    1. TASK_MODEL_URL environment variable
    2. settings.yaml servers.task_model_url
    3. settings.yaml llm.base_url (legacy)
    4. Default: http://localhost:8000/v1

    Args:
        settings: Pre-loaded settings dict, or None to load from file.

    Returns:
        Task model server URL.
    """
    # Environment variable takes precedence
    env_url = os.environ.get("TASK_MODEL_URL")
    if env_url:
        return env_url.rstrip("/")

    # Load settings if not provided
    if settings is None:
        settings = load_settings()

    # Check servers section first (new location)
    servers = settings.get("servers", {})
    if servers.get("task_model_url"):
        return servers["task_model_url"].rstrip("/")

    # Fall back to llm.base_url (legacy location)
    llm = settings.get("llm", {})
    if llm.get("base_url"):
        return llm["base_url"].rstrip("/")

    return DEFAULT_TASK_MODEL_URL


def get_genrm_url(settings: Optional[Dict[str, Any]] = None) -> str:
    """
    Get the GenRM (generative reward model) server URL.

    Priority:
    1. GENRM_URL environment variable
    2. settings.yaml servers.genrm_url
    3. Default: http://localhost:8001/v1

    Args:
        settings: Pre-loaded settings dict, or None to load from file.

    Returns:
        GenRM server URL.
    """
    # Environment variable takes precedence
    env_url = os.environ.get("GENRM_URL")
    if env_url:
        return env_url.rstrip("/")

    # Load settings if not provided
    if settings is None:
        settings = load_settings()

    # Check servers section
    servers = settings.get("servers", {})
    if servers.get("genrm_url"):
        return servers["genrm_url"].rstrip("/")

    return DEFAULT_GENRM_URL


def get_server_urls(settings: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Get all server URLs as a dict.

    Returns:
        Dict with 'task_model_url' and 'genrm_url' keys.
    """
    if settings is None:
        settings = load_settings()

    return {
        "task_model_url": get_task_model_url(settings),
        "genrm_url": get_genrm_url(settings),
    }


# Defaults - document_analysis is the general-purpose domain
# Use manifesto_rile for political manifesto scoring tasks
DEFAULT_DOMAIN = "document_analysis"
DEFAULT_TASK = "document_analysis"
DEFAULT_DATASET = "jsonl"  # Generic format, manifesto for manifesto-specific


def get_default_domain(settings: Optional[Dict[str, Any]] = None) -> str:
    """
    Get the default domain name.

    Priority:
    1. DOMAIN environment variable
    2. settings.yaml domains.default
    3. Default: manifesto_rile

    Args:
        settings: Pre-loaded settings dict, or None to load from file.

    Returns:
        Domain name.
    """
    # Environment variable takes precedence
    env_domain = os.environ.get("DOMAIN")
    if env_domain:
        return env_domain

    # Load settings if not provided
    if settings is None:
        settings = load_settings()

    # Check domains section
    domains = settings.get("domains", {})
    if domains.get("default"):
        return domains["default"]

    return DEFAULT_DOMAIN


def get_default_task(settings: Optional[Dict[str, Any]] = None) -> str:
    """
    Get the default task name.

    Priority:
    1. TASK environment variable
    2. settings.yaml tasks.default
    3. settings.yaml domains.default (legacy)
    4. Default: manifesto_rile
    """
    env_task = os.environ.get("TASK")
    if env_task:
        return env_task

    if settings is None:
        settings = load_settings()

    tasks = settings.get("tasks", {})
    if tasks.get("default"):
        return tasks["default"]

    return get_default_domain(settings)


def get_default_dataset(settings: Optional[Dict[str, Any]] = None) -> str:
    """
    Get the default dataset name.

    Priority:
    1. DATASET environment variable
    2. settings.yaml datasets.default
    3. Default: manifesto
    """
    env_dataset = os.environ.get("DATASET")
    if env_dataset:
        return env_dataset

    if settings is None:
        settings = load_settings()

    datasets = settings.get("datasets", {})
    if datasets.get("default"):
        return datasets["default"]

    return DEFAULT_DATASET


def get_domain_config(
    domain_name: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get configuration for a specific domain.

    Args:
        domain_name: Domain name (uses default if None)
        settings: Pre-loaded settings dict, or None to load from file.

    Returns:
        Domain configuration dict.
    """
    if settings is None:
        settings = load_settings()

    if domain_name is None:
        domain_name = get_default_domain(settings)

    domains = settings.get("domains", {})
    return domains.get(domain_name, {})


def get_task_config(
    task_name: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get configuration for a specific task."""
    if settings is None:
        settings = load_settings()

    if task_name is None:
        task_name = get_default_task(settings)

    tasks = settings.get("tasks", {})
    if task_name in tasks:
        return tasks.get(task_name, {})

    # Fall back to domains section (legacy)
    return get_domain_config(task_name, settings)


def get_dataset_config(
    dataset_name: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get configuration for a specific dataset."""
    if settings is None:
        settings = load_settings()

    if dataset_name is None:
        dataset_name = get_default_dataset(settings)

    datasets = settings.get("datasets", {})
    return datasets.get(dataset_name, {})
