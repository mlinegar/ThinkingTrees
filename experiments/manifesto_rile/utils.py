"""
Shared utilities for manifesto_rile experiments.

This module provides common setup functions used across experiment scripts.
"""

import sys
from pathlib import Path


def setup_python_path():
    """Add project root to Python path for imports to work."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


# Re-export centralized logging for convenience
from src.config.logging import setup_logging, get_logger


def print_banner(title: str, width: int = 60):
    """Print a formatted banner for experiment sections."""
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_section(title: str, char: str = "-", width: int = 40):
    """Print a section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")
