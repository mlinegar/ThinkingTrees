"""
Configuration module for the OPS framework.

This module provides centralized configuration for various aspects of the framework.
"""

from .concurrency import (
    ConcurrencyConfig,
    DEFAULT_CONCURRENCY,
    get_concurrency_config,
    create_low_resource_config,
    create_high_throughput_config,
)
from .settings import (
    default_settings_path,
    load_settings,
    get_task_model_url,
    get_genrm_url,
    get_server_urls,
    get_default_domain,
    get_default_task,
    get_default_dataset,
    get_domain_config,
    get_task_config,
    get_dataset_config,
    DEFAULT_TASK_MODEL_URL,
    DEFAULT_GENRM_URL,
    DEFAULT_DOMAIN,
    DEFAULT_TASK,
    DEFAULT_DATASET,
)
from .dspy_config import (
    get_xml_adapter,
    configure_dspy,
)

__all__ = [
    'ConcurrencyConfig',
    'DEFAULT_CONCURRENCY',
    'get_concurrency_config',
    'create_low_resource_config',
    'create_high_throughput_config',
    'default_settings_path',
    'load_settings',
    'get_task_model_url',
    'get_genrm_url',
    'get_server_urls',
    'get_default_domain',
    'get_default_task',
    'get_default_dataset',
    'get_domain_config',
    'get_task_config',
    'get_dataset_config',
    'DEFAULT_TASK_MODEL_URL',
    'DEFAULT_GENRM_URL',
    'DEFAULT_DOMAIN',
    'DEFAULT_TASK',
    'DEFAULT_DATASET',
    'get_xml_adapter',
    'configure_dspy',
]
