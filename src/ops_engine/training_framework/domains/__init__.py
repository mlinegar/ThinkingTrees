"""
Domain plugins submodule for the training framework.

This module provides a plugin system for domain-specific training integration.
Each domain represents a different use case (e.g., manifesto/RILE, legal,
sentiment) that can plug into the OPS training framework.

Supported Domains:
- manifesto_rile: Political manifesto RILE scoring

Usage:
    from ops_engine.training_framework.domains import (
        DomainRegistry,
        get_domain,
        list_domains,
    )

    # Get a domain instance
    domain = get_domain("manifesto_rile", bin_size=10.0)

    # Create domain-specific components
    metric = domain.create_metric(weighted=True)
    classifier = domain.create_classifier()
    training_source = domain.create_training_source(results)

    # List available domains
    available = list_domains()
"""

from typing import Any, Dict, Optional

# Import base classes
from .base import (
    DomainPlugin,
    AbstractDomain,
)

# Import registry
from .registry import (
    DomainRegistry,
    register_domain,
)

# Import domain implementations (registration happens on import)
from .manifesto import ManifestoDomain


# =============================================================================
# Convenience Functions
# =============================================================================

def get_domain(name: str, **kwargs) -> DomainPlugin:
    """
    Get a domain instance by name.

    Args:
        name: Domain name (e.g., 'manifesto_rile')
        **kwargs: Arguments to pass to domain constructor

    Returns:
        Domain instance

    Raises:
        KeyError: If domain not found
    """
    return DomainRegistry.get(name, **kwargs)


def list_domains() -> Dict[str, Dict[str, Any]]:
    """
    List all registered domains.

    Returns:
        Dict mapping domain names to metadata
    """
    return DomainRegistry.list_domains()


def is_domain_registered(name: str) -> bool:
    """
    Check if a domain is registered.

    Args:
        name: Domain name

    Returns:
        True if registered, False otherwise
    """
    return DomainRegistry.is_registered(name)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Base classes
    'DomainPlugin',
    'AbstractDomain',

    # Registry
    'DomainRegistry',
    'register_domain',

    # Domain implementations
    'ManifestoDomain',

    # Convenience functions
    'get_domain',
    'list_domains',
    'is_domain_registered',
]
