"""
Unified OPS Law Checking Utilities.

This module provides the core logic for checking OPS (Oracle-Preserving Summarization)
laws, designed to be used by both Auditor and OracleNodeVerifier to prevent
implementation drift.

Laws implemented:
- C1 (Sufficiency): g(b) ∼ b — leaf summary preserves oracle value
- C2 (Idempotence): g(s) ∼ s — re-summarizing doesn't change oracle
- C3A (Substitution): Equivalent summaries at boundary → same oracle
- C3B (Merge Consistency): u ⊕ v ∼ g(u ⊕ v) ∼ g(g(u) ⊕ g(v))

Usage:
    from src.ops_engine.checks import CheckRunner, CheckConfig, CheckResult

    config = CheckConfig(discrepancy_threshold=0.1)
    runner = CheckRunner(oracle_fn=my_oracle, config=config)

    result = runner.check_sufficiency(original_text, summary, rubric)
    if not result.passed:
        print(f"Sufficiency violation: {result.discrepancy}")
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List, Dict, Any, Protocol, Union
from enum import Enum
import logging

from src.ops_engine.scoring import ScoringOracle, OracleScore
from src.core.protocols import format_merge_input

logger = logging.getLogger(__name__)


class CheckType(Enum):
    """Types of OPS law checks."""
    SUFFICIENCY = "sufficiency"      # C1: Leaf summary preserves oracle
    IDEMPOTENCE = "idempotence"      # C2: Re-summarizing is stable
    SUBSTITUTION = "substitution"    # C3A: Boundary consistency
    MERGE = "merge_consistency"      # C3B: Merge preserves oracle


@dataclass
class CheckConfig:
    """Configuration for OPS law checks."""

    # Discrepancy threshold - scores above this are violations
    discrepancy_threshold: float = 0.1

    # Whether to treat close values as equivalent (for ordinal oracles)
    tolerance: float = 0.0


@dataclass
class CheckResult:
    """
    Result of an OPS law check.

    This unified format is used by all check types and can be converted
    to both AuditCheckResult (for Auditor) and LawCheckResult (for OracleNodeVerifier).
    """
    check_type: CheckType
    passed: bool
    discrepancy: float
    reasoning: str

    # The values that were compared
    value_a: Optional[Any] = None  # e.g., oracle(original)
    value_b: Optional[Any] = None  # e.g., oracle(summary)

    # Raw scores if available
    score_a: Optional[float] = None
    score_b: Optional[float] = None

    # Node/context info
    node_id: Optional[str] = None

    # Skipped checks (e.g., no summarizer provided)
    # IMPORTANT: skipped checks are NOT passed checks - they couldn't be evaluated
    skipped: bool = False
    skip_reason: Optional[str] = None

    @property
    def is_violation(self) -> bool:
        """Alias for not passed. Skipped checks are not violations."""
        return not self.passed and not self.skipped

    @property
    def was_evaluated(self) -> bool:
        """True if the check was actually performed (not skipped)."""
        return not self.skipped

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'check_type': self.check_type.value,
            'passed': self.passed,
            'discrepancy': self.discrepancy,
            'reasoning': self.reasoning,
            'value_a': str(self.value_a) if self.value_a else None,
            'value_b': str(self.value_b) if self.value_b else None,
            'score_a': self.score_a,
            'score_b': self.score_b,
            'node_id': self.node_id,
            'skipped': self.skipped,
            'skip_reason': self.skip_reason,
        }


class OracleProtocol(Protocol):
    """
    Protocol for oracle functions used in OPS checks.

    An oracle takes two texts and a rubric, and returns:
    - is_congruent: bool - whether the texts produce equivalent oracle values
    - discrepancy: float - how different the oracle values are (0 = same)
    - reasoning: str - explanation of the comparison
    """

    def __call__(
        self,
        input_a: str,
        input_b: str,
        rubric: str,
    ) -> Tuple[bool, float, str]:
        """Compare two inputs and return (is_congruent, discrepancy, reasoning)."""
        ...


class CheckRunner:
    """
    Runs OPS law checks using a provided oracle function.

    This class provides the core check logic that can be used by both
    Auditor and OracleNodeVerifier, ensuring consistent behavior.

    Accepts both ScoringOracle (preferred) and legacy OracleProtocol.
    """

    def __init__(
        self,
        oracle_fn: Union[ScoringOracle, OracleProtocol],
        config: Optional[CheckConfig] = None,
        summarizer_fn: Optional[Callable[[str, str], str]] = None,
    ):
        """
        Initialize the check runner.

        Args:
            oracle_fn: Oracle implementing ScoringOracle protocol
            config: Check configuration (uses defaults if None)
            summarizer_fn: Optional summarizer for idempotence checks
                          Signature: (content, rubric) -> summary
        """
        self._oracle = oracle_fn
        self.config = config or CheckConfig()
        self.summarizer = summarizer_fn

    def _call_oracle(self, input_a: str, input_b: str, rubric: str) -> Tuple[bool, float, str]:
        """
        Call the oracle and return (is_congruent, discrepancy, reasoning).

        Requires ScoringOracle interface (has .score() method returning OracleScore).
        """
        result = self._oracle.score(input_a, input_b, rubric)
        # Score is 0.0-1.0 where 1.0 = good; discrepancy is inverse
        discrepancy = 1.0 - result.score
        is_congruent = discrepancy <= self.config.discrepancy_threshold
        return is_congruent, discrepancy, result.reasoning

    def check_sufficiency(
        self,
        original_content: str,
        summary: str,
        rubric: str,
        node_id: Optional[str] = None,
    ) -> CheckResult:
        """
        Check C1 (Sufficiency): Does the summary preserve oracle-relevant info?

        Compares oracle(original) with oracle(summary). If they differ beyond
        the threshold, the summary has lost information needed to compute the oracle.

        Per paper Section 3.2:
            "A summarizer g is sufficient for oracle f* if, for all inputs b,
            oracle(g(b)) ≈ oracle(b)"

        Args:
            original_content: The original text (block b)
            summary: The summary g(b)
            rubric: Description of what to preserve
            node_id: Optional node identifier

        Returns:
            CheckResult with pass/fail and discrepancy
        """
        is_congruent, discrepancy, reasoning = self._call_oracle(
            original_content,
            summary,
            rubric,
        )

        passed = is_congruent and discrepancy <= self.config.discrepancy_threshold

        if not passed:
            reasoning = (
                f"C1 Sufficiency violation: Original and summary produce different "
                f"oracle values (discrepancy={discrepancy:.4f}, threshold={self.config.discrepancy_threshold}). "
                f"{reasoning}"
            )

        return CheckResult(
            check_type=CheckType.SUFFICIENCY,
            passed=passed,
            discrepancy=discrepancy,
            reasoning=reasoning,
            value_a=original_content,
            value_b=summary,
            node_id=node_id,
        )

    def check_idempotence(
        self,
        summary: str,
        rubric: str,
        re_summary: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> CheckResult:
        """
        Check C2 (Idempotence): Does re-summarizing change the oracle?

        A stable summary should produce the same oracle prediction even after
        being summarized again.

        Per paper Section 3.3:
            "A summarizer g is idempotent if g(s) ≈ s for all s in range(g)"

        Args:
            summary: The summary to check
            rubric: Description of what to preserve
            re_summary: Pre-computed g(summary). If None, uses self.summarizer
            node_id: Optional node identifier

        Returns:
            CheckResult with pass/fail and discrepancy
        """
        # Get re-summary if not provided
        if re_summary is None:
            if self.summarizer is None:
                return CheckResult(
                    check_type=CheckType.IDEMPOTENCE,
                    passed=False,  # NOT passed - check wasn't performed
                    discrepancy=0.0,
                    reasoning="Skipped: no summarizer provided for idempotence check",
                    node_id=node_id,
                    skipped=True,
                    skip_reason="no_summarizer",
                )
            re_summary = self.summarizer(summary, rubric)

        is_congruent, discrepancy, reasoning = self._call_oracle(
            summary,
            re_summary,
            rubric,
        )

        passed = is_congruent and discrepancy <= self.config.discrepancy_threshold

        if not passed:
            reasoning = (
                f"C2 Idempotence violation: Re-summarizing changes oracle "
                f"(discrepancy={discrepancy:.4f}). Original may contain extraneous detail. "
                f"{reasoning}"
            )

        return CheckResult(
            check_type=CheckType.IDEMPOTENCE,
            passed=passed,
            discrepancy=discrepancy,
            reasoning=reasoning,
            value_a=summary,
            value_b=re_summary,
            node_id=node_id,
        )

    def check_substitution(
        self,
        left_raw: str,
        right_raw: str,
        left_summary: str,
        right_summary: str,
        rubric: str,
        node_id: Optional[str] = None,
    ) -> CheckResult:
        """
        Check C3A (Substitution): Leaf boundary consistency.

        At a leaf boundary, g(left ⊕ right) should be equivalent to
        g(g(left) ⊕ g(right)).

        Per paper Section 3.4:
            "Substitution consistency requires that equivalent summaries
            produce the same oracle prediction when merged."

        Args:
            left_raw: Raw text of left leaf
            right_raw: Raw text of right leaf
            left_summary: Summary of left leaf g(left)
            right_summary: Summary of right leaf g(right)
            rubric: Description of what to preserve
            node_id: Optional node identifier

        Returns:
            CheckResult with pass/fail and discrepancy
        """
        # Concatenate raw and summaries
        joint_raw = format_merge_input(left_raw, right_raw)
        joint_summaries = format_merge_input(left_summary, right_summary)

        # Get summaries of both concatenations
        if self.summarizer is None:
            return CheckResult(
                check_type=CheckType.SUBSTITUTION,
                passed=False,  # NOT passed - check wasn't performed
                discrepancy=0.0,
                reasoning="Skipped: no summarizer provided for substitution check",
                node_id=node_id,
                skipped=True,
                skip_reason="no_summarizer",
            )

        summary_of_raw = self.summarizer(joint_raw, rubric)
        summary_of_summaries = self.summarizer(joint_summaries, rubric)

        # Compare oracle predictions
        is_congruent, discrepancy, reasoning = self._call_oracle(
            summary_of_raw,
            summary_of_summaries,
            rubric,
        )

        passed = is_congruent and discrepancy <= self.config.discrepancy_threshold

        if not passed:
            reasoning = (
                f"C3A Substitution violation: g(raw_left ⊕ raw_right) ≠ g(g(left) ⊕ g(right)) "
                f"(discrepancy={discrepancy:.4f}). {reasoning}"
            )

        return CheckResult(
            check_type=CheckType.SUBSTITUTION,
            passed=passed,
            discrepancy=discrepancy,
            reasoning=reasoning,
            value_a=summary_of_raw,
            value_b=summary_of_summaries,
            node_id=node_id,
        )

    def check_merge_consistency(
        self,
        child_summaries: List[str],
        parent_summary: str,
        rubric: str,
        node_id: Optional[str] = None,
    ) -> CheckResult:
        """
        Check C3B (Merge Consistency): Does merge preserve oracle info?

        Compares the parent's summary with the concatenation of child summaries.
        The oracle should be consistent whether computed from children or parent.

        Per paper Section 3.4:
            "Merge consistency requires that merging child summaries and then
            summarizing produces the same oracle as directly summarizing."

        Args:
            child_summaries: Summaries of child nodes
            parent_summary: Summary of the parent (internal) node
            rubric: Description of what to preserve
            node_id: Optional node identifier

        Returns:
            CheckResult with pass/fail and discrepancy
        """
        # Concatenate child summaries
        children_concat = format_merge_input(*child_summaries)

        is_congruent, discrepancy, reasoning = self._call_oracle(
            children_concat,
            parent_summary,
            rubric,
        )

        passed = is_congruent and discrepancy <= self.config.discrepancy_threshold

        if not passed:
            reasoning = (
                f"C3B Merge violation: oracle(children) ≠ oracle(parent) "
                f"(discrepancy={discrepancy:.4f}). Information lost in merge. {reasoning}"
            )

        return CheckResult(
            check_type=CheckType.MERGE,
            passed=passed,
            discrepancy=discrepancy,
            reasoning=reasoning,
            value_a=children_concat,
            value_b=parent_summary,
            node_id=node_id,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def run_all_checks(
    runner: CheckRunner,
    original_content: str,
    summary: str,
    rubric: str,
    child_summaries: Optional[List[str]] = None,
    is_leaf: bool = True,
    node_id: Optional[str] = None,
) -> List[CheckResult]:
    """
    Run all applicable OPS checks for a node.

    Args:
        runner: The CheckRunner to use
        original_content: Original text (for leaf) or None
        summary: The node's summary
        rubric: Preservation rubric
        child_summaries: Child summaries (for internal nodes)
        is_leaf: Whether this is a leaf node
        node_id: Optional node identifier

    Returns:
        List of CheckResult for all applicable checks
    """
    results = []

    if is_leaf and original_content:
        # Leaf: check sufficiency
        results.append(runner.check_sufficiency(
            original_content, summary, rubric, node_id
        ))

    if summary:
        # All nodes: check idempotence
        results.append(runner.check_idempotence(
            summary, rubric, node_id=node_id
        ))

    if not is_leaf and child_summaries:
        # Internal: check merge consistency
        results.append(runner.check_merge_consistency(
            child_summaries, summary, rubric, node_id
        ))

    return results


def aggregate_check_stats(results: List[CheckResult]) -> Dict[str, Any]:
    """
    Aggregate statistics from multiple check results.

    Returns:
        Dict with violation counts and rates per check type
    """
    stats = {
        "total_checks": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "by_type": {},
    }

    for check_type in CheckType:
        type_results = [r for r in results if r.check_type == check_type]
        if type_results:
            n_passed = sum(1 for r in type_results if r.passed)
            n_total = len(type_results)
            stats["by_type"][check_type.value] = {
                "total": n_total,
                "passed": n_passed,
                "failed": n_total - n_passed,
                "pass_rate": n_passed / n_total if n_total > 0 else 1.0,
            }

    return stats
