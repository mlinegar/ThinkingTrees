"""
Domain registry for managing domain plugins.

This module provides a central registry for domain implementations,
allowing dynamic discovery and selection of domains.
"""

import logging
from typing import Dict, List, Optional, Type

from .base import DomainPlugin, AbstractDomain

logger = logging.getLogger(__name__)


class DomainRegistry:
    """
    Registry for domain implementations.

    Provides a central place to register and retrieve domain plugins.
    Domains can be registered by name and retrieved dynamically.
    """

    _domains: Dict[str, Type[DomainPlugin]] = {}
    _instances: Dict[str, DomainPlugin] = {}

    @classmethod
    def register(cls, name: str, domain_class: Type[DomainPlugin]) -> None:
        """
        Register a domain implementation.

        Args:
            name: Unique name for the domain
            domain_class: Domain class implementing DomainPlugin
        """
        if name in cls._domains:
            logger.warning(f"Overwriting existing domain: {name}")
        cls._domains[name] = domain_class
        logger.debug(f"Registered domain: {name}")

    @classmethod
    def get_class(cls, name: str) -> Type[DomainPlugin]:
        """
        Get domain class by name.

        Args:
            name: Domain name

        Returns:
            Domain class

        Raises:
            KeyError: If domain not found
        """
        if name not in cls._domains:
            available = list(cls._domains.keys())
            raise KeyError(f"Unknown domain: {name}. Available: {available}")
        return cls._domains[name]

    @classmethod
    def get(cls, name: str, **kwargs) -> DomainPlugin:
        """
        Get or create a domain instance.

        Args:
            name: Domain name
            **kwargs: Arguments to pass to domain constructor

        Returns:
            Domain instance
        """
        # Create new instance with provided kwargs
        domain_class = cls.get_class(name)
        return domain_class(**kwargs)

    @classmethod
    def get_singleton(cls, name: str, **kwargs) -> DomainPlugin:
        """
        Get a singleton domain instance (cached).

        Args:
            name: Domain name
            **kwargs: Arguments to pass to domain constructor (only used on first call)

        Returns:
            Cached domain instance
        """
        if name not in cls._instances:
            cls._instances[name] = cls.get(name, **kwargs)
        return cls._instances[name]

    @classmethod
    def list_domains(cls) -> Dict[str, Dict]:
        """
        List all registered domains with their metadata.

        Returns:
            Dict mapping names to domain metadata
        """
        result = {}
        for name, domain_class in cls._domains.items():
            try:
                # Try to get info from class
                result[name] = {
                    'class': domain_class.__name__,
                    'module': domain_class.__module__,
                }
            except Exception:
                result[name] = {
                    'class': str(domain_class),
                    'module': 'unknown',
                }
        return result

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a domain is registered."""
        return name in cls._domains

    @classmethod
    def clear(cls) -> None:
        """Clear all registered domains (mainly for testing)."""
        cls._domains.clear()
        cls._instances.clear()


def register_domain(name: str):
    """
    Decorator to register a domain class.

    Usage:
        @register_domain("my_domain")
        class MyDomain(AbstractDomain):
            ...
    """
    def decorator(cls: Type[DomainPlugin]) -> Type[DomainPlugin]:
        DomainRegistry.register(name, cls)
        return cls
    return decorator
