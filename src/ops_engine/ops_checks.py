"""
Unified OPS Law Checking Types and Utilities.

This module provides shared types for OPS (Oracle-Preserving Summarization)
law checking, used by both the Auditor (human-in-the-loop) and Verification
(training data generation) systems.

OPS Laws:
- C1 (Sufficiency): g(b) ∼ b — leaf summary preserves oracle value
- C2 (Idempotence): g(s) ∼ s — re-summarizing doesn't change oracle
- C3A (Substitution): Equivalent summaries at boundary → same oracle
- C3B (Merge Consistency): u ⊕ v ∼ g(u ⊕ v) ∼ g(g(u) ⊕ g(v))

Usage:
    from src.ops_engine.ops_checks import CheckType, CheckConfig

    # For human-in-the-loop auditing:
    from src.ops_engine.auditor import Auditor, AuditConfig, ReviewQueue

    # For training data generation:
    from src.ops_engine.training_framework.verification import TreeVerifier
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class CheckType(Enum):
    """Types of OPS law checks.

    These correspond to the four conditions from the OPS paper:
    - SUFFICIENCY (C1): Leaf summary preserves oracle-relevant information
    - IDEMPOTENCE (C2): Re-summarizing is stable (g(g(x)) ≈ g(x))
    - SUBSTITUTION (C3A): Boundary consistency for adjacent leaves
    - MERGE (C3B): Merging children preserves oracle information
    """
    SUFFICIENCY = "sufficiency"      # C1: Leaf summary preserves oracle
    IDEMPOTENCE = "idempotence"      # C2: Re-summarizing is stable
    SUBSTITUTION = "substitution"    # C3A: Boundary consistency
    MERGE = "merge_consistency"      # C3B: Merge preserves oracle

    @classmethod
    def from_string(cls, s: str) -> "CheckType":
        """Convert string to CheckType, handling various formats."""
        normalized = s.lower().strip()
        if normalized in ("merge", "merge_consistency"):
            return cls.MERGE
        for check_type in cls:
            if check_type.value == normalized:
                return check_type
        raise ValueError(f"Unknown check type: {s}")

    def __str__(self) -> str:
        return self.value


@dataclass
class CheckConfig:
    """Configuration for OPS law checks.

    This provides a unified configuration interface for check parameters
    that can be used by both Auditor and Verifier implementations.
    """
    # Discrepancy threshold - scores above this are violations
    discrepancy_threshold: float = 0.1

    # Whether to treat close values as equivalent (for ordinal oracles)
    tolerance: float = 0.0

    # Which checks to enable
    check_sufficiency: bool = True
    check_idempotence: bool = True
    check_substitution: bool = True
    check_merge: bool = True


@dataclass
class CheckResult:
    """
    Unified result format for OPS law checks.

    This is a base result type that can be used across different
    check implementations. Both AuditCheckResult and LawCheckResult
    are compatible with this interface.
    """
    check_type: CheckType
    passed: bool
    discrepancy: float
    reasoning: str = ""

    # The values that were compared (optional)
    input_a: Optional[str] = None
    input_b: Optional[str] = None

    # Node/context info
    node_id: Optional[str] = None

    # Skipped checks (e.g., no summarizer provided)
    skipped: bool = False
    skip_reason: Optional[str] = None

    @property
    def is_violation(self) -> bool:
        """True if this is a violation (failed and not skipped)."""
        return not self.passed and not self.skipped

    @property
    def was_evaluated(self) -> bool:
        """True if the check was actually performed (not skipped)."""
        return not self.skipped

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'check_type': str(self.check_type),
            'passed': self.passed,
            'discrepancy': self.discrepancy,
            'reasoning': self.reasoning,
            'input_a': self.input_a[:100] + '...' if self.input_a and len(self.input_a) > 100 else self.input_a,
            'input_b': self.input_b[:100] + '...' if self.input_b and len(self.input_b) > 100 else self.input_b,
            'node_id': self.node_id,
            'skipped': self.skipped,
            'skip_reason': self.skip_reason,
        }


def aggregate_check_stats(results: List[CheckResult]) -> Dict[str, Any]:
    """
    Aggregate statistics from multiple check results.

    Args:
        results: List of CheckResult objects

    Returns:
        Dict with violation counts and rates per check type
    """
    stats = {
        "total_checks": len(results),
        "total_passed": sum(1 for r in results if r.passed),
        "total_failed": sum(1 for r in results if not r.passed and not r.skipped),
        "total_skipped": sum(1 for r in results if r.skipped),
        "by_type": {},
    }

    for check_type in CheckType:
        type_results = [r for r in results if r.check_type == check_type]
        if type_results:
            n_passed = sum(1 for r in type_results if r.passed)
            n_failed = sum(1 for r in type_results if not r.passed and not r.skipped)
            n_skipped = sum(1 for r in type_results if r.skipped)
            n_total = len(type_results)
            n_evaluated = n_total - n_skipped
            stats["by_type"][str(check_type)] = {
                "total": n_total,
                "passed": n_passed,
                "failed": n_failed,
                "skipped": n_skipped,
                "pass_rate": n_passed / n_evaluated if n_evaluated > 0 else 1.0,
                "violation_rate": n_failed / n_evaluated if n_evaluated > 0 else 0.0,
            }

    # Overall rates
    n_evaluated = stats["total_checks"] - stats["total_skipped"]
    stats["overall_pass_rate"] = stats["total_passed"] / n_evaluated if n_evaluated > 0 else 1.0
    stats["overall_violation_rate"] = stats["total_failed"] / n_evaluated if n_evaluated > 0 else 0.0

    return stats


# =============================================================================
# Protocol for Oracle Functions
# =============================================================================

@runtime_checkable
class OracleProtocol(Protocol):
    """
    Protocol for oracle functions used in OPS law checking.

    An oracle compares two texts and determines if they are congruent
    with respect to preserving task-relevant information.
    """

    def __call__(
        self,
        input_a: str,
        input_b: str,
        rubric: str,
    ) -> tuple:
        """
        Compare two inputs and return congruence result.

        Args:
            input_a: First input text
            input_b: Second input text
            rubric: Description of what to preserve

        Returns:
            Tuple of (is_congruent: bool, discrepancy: float, reasoning: str)
        """
        ...


# Convenience re-exports
__all__ = [
    "CheckType",
    "CheckConfig",
    "CheckResult",
    "aggregate_check_stats",
    "OracleProtocol",
]
