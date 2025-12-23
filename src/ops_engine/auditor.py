"""
OPS Auditor - Probabilistic verification of summarization quality.

The auditor samples nodes from the OPS tree and verifies that summaries
preserve the information specified in the rubric. It uses an Oracle
(approximate or exact) to detect information loss.

Key features:
- Probabilistic sampling with configurable budget
- Flagging system for human/oracle batch review
- Review queue for collecting items needing verification
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Protocol, Set, Tuple, Dict, Any, Union
from enum import Enum
from datetime import datetime
import random
import logging
import json
import warnings

from src.core.data_models import OPSNode, OPSTree, AuditStatus, AuditResult
from src.ops_engine.scoring import OracleScore, ScoringOracle, LegacyOracleAdapter
from src.config.concurrency import ConcurrencyConfig, get_concurrency_config


logger = logging.getLogger(__name__)


class OracleJudge(Protocol):
    """Protocol for oracle judge functions."""

    def __call__(
        self,
        input_a: str,
        input_b: str,
        rubric: str
    ) -> Tuple[bool, float, str]:
        """
        Compare two inputs for task-equivalence according to rubric.

        Args:
            input_a: First input (e.g., concatenated child summaries)
            input_b: Second input (e.g., parent summary)
            rubric: Information preservation criteria

        Returns:
            Tuple of:
            - is_congruent: bool - Are the inputs task-equivalent?
            - discrepancy_score: float - 0.0 (perfect) to 1.0 (total loss)
            - reasoning: str - Explanation of the comparison
        """
        ...


# =============================================================================
# Score-Centric Oracles (New, Preferred API)
# =============================================================================

class SimpleScorer:
    """
    Simple word-overlap scorer implementing ScoringOracle.

    This is the preferred API for new code. Returns OracleScore with
    score (1.0 = good) as primary output.

    Example:
        scorer = SimpleScorer()
        result = scorer.score(text_a, text_b, rubric)
        print(result.score)  # 0.85
        print(result.passes_threshold(0.8))  # True
    """

    def score(
        self,
        input_a: str,
        input_b: str,
        rubric: str,
    ) -> OracleScore:
        """
        Score similarity between two inputs using word overlap.

        Args:
            input_a: First input text
            input_b: Second input text
            rubric: Comparison criteria (unused in simple implementation)

        Returns:
            OracleScore with similarity score (1.0 = identical)
        """
        words_a = set(input_a.lower().split())
        words_b = set(input_b.lower().split())

        if not words_a or not words_b:
            return OracleScore(
                score=1.0,
                reasoning="Empty input(s)",
            )

        intersection = words_a & words_b
        union = words_a | words_b

        similarity = len(intersection) / len(union) if union else 1.0

        return OracleScore(
            score=similarity,
            reasoning=f"Word overlap: {similarity:.2%}",
        )


# =============================================================================
# Oracle Adapters
# =============================================================================

def create_oracle_from_scorer(
    scorer: ScoringOracle,
    threshold: float = 0.1,
) -> Callable[[str, str, str], Tuple[bool, float, str]]:
    """
    Create a legacy oracle callable from a ScoringOracle.

    This adapter allows using the new ScoringOracle API with code that
    expects the legacy (bool, float, str) tuple format.

    Args:
        scorer: A ScoringOracle instance (e.g., SimpleScorer)
        threshold: Discrepancy threshold for congruence (0.0-1.0)

    Returns:
        Callable with signature (input_a, input_b, rubric) -> (is_congruent, discrepancy, reasoning)

    Example:
        from src.ops_engine.scoring import SimpleScorer
        scorer = SimpleScorer()
        oracle = create_oracle_from_scorer(scorer, threshold=0.1)

        # Now use with OPSAuditor
        auditor = OPSAuditor(oracle=oracle, config=config)
    """
    def oracle_fn(input_a: str, input_b: str, rubric: str) -> Tuple[bool, float, str]:
        result = scorer.score(input_a, input_b, rubric)
        discrepancy = result.to_discrepancy()
        is_congruent = discrepancy <= threshold
        return is_congruent, discrepancy, result.reasoning

    return oracle_fn


class AlwaysPassOracle:
    """Oracle that always passes - useful for testing."""

    def __call__(
        self,
        input_a: str,
        input_b: str,
        rubric: str
    ) -> Tuple[bool, float, str]:
        return True, 0.0, "Always pass oracle"


class AlwaysFailOracle:
    """Oracle that always fails - useful for testing."""

    def __call__(
        self,
        input_a: str,
        input_b: str,
        rubric: str
    ) -> Tuple[bool, float, str]:
        return False, 1.0, "Always fail oracle"


class SamplingStrategy(Enum):
    """Strategy for selecting nodes to audit."""
    RANDOM = "random"              # Uniform random sampling
    LEVEL_WEIGHTED = "level_weighted"  # Prefer higher levels (more compression)
    PRIORITY = "priority"          # Use node priority scores


class Summarizer(Protocol):
    """Protocol for summarizer functions used in idempotence/substitution checks."""

    def __call__(self, text: str, rubric: str) -> str:
        """
        Summarize the given text according to the rubric.

        Args:
            text: Input text to summarize
            rubric: Information preservation criteria

        Returns:
            Summary string
        """
        ...


@dataclass
class AuditConfig:
    """Configuration for the auditor."""

    # Sampling parameters
    sample_budget: int = 10
    sampling_strategy: SamplingStrategy = SamplingStrategy.RANDOM
    sampling_probability: float = 1.0  # For probabilistic sampling

    # Thresholds
    discrepancy_threshold: float = 0.1

    # Flags
    audit_leaves: bool = True
    audit_internal: bool = True
    prioritize_high_levels: bool = True

    # Idempotence and substitution checks (from paper Section 4.1)
    audit_idempotence: bool = True  # Check if re-summarizing summaries preserves oracle (C2)
    audit_substitution: bool = True  # Check leaf boundary substitution consistency (C3 Case A)
    idempotence_budget: int = 5  # Number of summaries to sample for idempotence check
    substitution_budget: int = 5  # Number of leaf boundaries to sample for substitution check

    # Seed for reproducibility
    random_seed: Optional[int] = None

    # Concurrency settings (uses centralized config)
    concurrency: Optional[ConcurrencyConfig] = None

    def get_concurrency(self) -> ConcurrencyConfig:
        """Get concurrency config, using default if not set."""
        return self.concurrency or get_concurrency_config()


@dataclass
class AuditCheckResult:
    """Result of a single audit check."""
    node_id: str
    check_type: str  # "sufficiency" or "merge_consistency"
    passed: bool
    discrepancy_score: float
    reasoning: str
    input_a: str = ""
    input_b: str = ""


@dataclass
class AuditReport:
    """Complete audit report for a tree."""
    tree_id: str
    total_nodes: int
    nodes_audited: int
    nodes_passed: int
    nodes_failed: int
    failure_rate: float
    checks: List[AuditCheckResult] = field(default_factory=list)
    failed_node_ids: List[str] = field(default_factory=list)

    # Violation rates by check type (from paper Section 4.1)
    sufficiency_violations: int = 0  # p_suff: leaf sufficiency failures
    merge_violations: int = 0  # p_merge: internal merge failures
    idempotence_violations: int = 0  # p_idem: idempotence failures (C2)
    substitution_violations: int = 0  # p_bound: leaf boundary substitution failures

    # Sample counts for computing rates
    sufficiency_samples: int = 0
    merge_samples: int = 0
    idempotence_samples: int = 0
    substitution_samples: int = 0

    @property
    def passed(self) -> bool:
        """Overall audit passed (no failures)."""
        return self.nodes_failed == 0

    @property
    def sufficiency_rate(self) -> float:
        """Empirical sufficiency violation rate (p_suff)."""
        return self.sufficiency_violations / self.sufficiency_samples if self.sufficiency_samples > 0 else 0.0

    @property
    def merge_rate(self) -> float:
        """Empirical merge violation rate (p_merge)."""
        return self.merge_violations / self.merge_samples if self.merge_samples > 0 else 0.0

    @property
    def idempotence_rate(self) -> float:
        """Empirical idempotence violation rate (p_idem)."""
        return self.idempotence_violations / self.idempotence_samples if self.idempotence_samples > 0 else 0.0

    @property
    def substitution_rate(self) -> float:
        """Empirical substitution violation rate (p_bound)."""
        return self.substitution_violations / self.substitution_samples if self.substitution_samples > 0 else 0.0

    @property
    def assoc_rate(self) -> float:
        """
        Combined merge-consistency violation rate (p_assoc).

        Weighted average of substitution (leaf boundary) and merge (internal) rates.
        From paper: p_assoc = λ * p_bound + (1-λ) * p_merge
        """
        total = self.substitution_samples + self.merge_samples
        if total == 0:
            return 0.0
        lambda_weight = self.substitution_samples / total
        return lambda_weight * self.substitution_rate + (1 - lambda_weight) * self.merge_rate


class ReviewPriority(Enum):
    """Priority levels for review items."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class FlaggedItem:
    """
    An item flagged for human or oracle review.

    Contains all information needed for batch processing of reviews.
    """
    # Identity
    item_id: str
    node_id: str
    tree_id: str

    # Check details
    check_type: str  # "sufficiency" or "merge_consistency"
    input_a: str     # Original/source content
    input_b: str     # Summary/target content
    rubric: str      # Information preservation criteria

    # Audit results from approximate oracle
    approx_discrepancy: float
    approx_reasoning: str

    # Metadata
    priority: ReviewPriority = ReviewPriority.MEDIUM
    flagged_at: str = field(default_factory=lambda: datetime.now().isoformat())
    node_level: int = 0

    # Review results (filled in after human/oracle review)
    reviewed: bool = False
    review_result: Optional[bool] = None  # True = approved, False = needs fix
    review_reasoning: Optional[str] = None
    corrected_summary: Optional[str] = None
    reviewed_at: Optional[str] = None
    review_source: str = "human"  # "human" or "oracle_func_auto" - use to filter training data

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "item_id": self.item_id,
            "node_id": self.node_id,
            "tree_id": self.tree_id,
            "check_type": self.check_type,
            "input_a": self.input_a,
            "input_b": self.input_b,
            "rubric": self.rubric,
            "approx_discrepancy": self.approx_discrepancy,
            "approx_reasoning": self.approx_reasoning,
            "priority": self.priority.name,
            "flagged_at": self.flagged_at,
            "node_level": self.node_level,
            "reviewed": self.reviewed,
            "review_result": self.review_result,
            "review_reasoning": self.review_reasoning,
            "corrected_summary": self.corrected_summary,
            "reviewed_at": self.reviewed_at,
            "review_source": self.review_source
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FlaggedItem':
        """Create from dictionary."""
        data = dict(data)
        data["priority"] = ReviewPriority[data.get("priority", "MEDIUM")]
        return cls(**data)


class ReviewQueue:
    """
    Queue for collecting flagged items for batch human/oracle review.

    Supports:
    - Adding flagged items from audit failures
    - Prioritization by level and discrepancy score
    - Batch export for processing
    - Import of review results

    Example:
        >>> queue = ReviewQueue()
        >>> auditor = OPSAuditor(oracle, config, review_queue=queue)
        >>> auditor.audit_tree(tree)
        >>> batch = queue.get_batch(limit=10)
        >>> # Process batch with human reviewers or exact oracle
        >>> for item in batch:
        ...     item.reviewed = True
        ...     item.review_result = True  # Approved
        >>> queue.import_results(batch)
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize review queue.

        Args:
            max_size: Maximum items to hold in queue
        """
        self.max_size = max_size
        self._items: Dict[str, FlaggedItem] = {}
        self._item_counter = 0

    def add(
        self,
        node: OPSNode,
        tree_id: str,
        check_result: AuditCheckResult,
        rubric: str,
        full_input_a: str = "",
        full_input_b: str = ""
    ) -> FlaggedItem:
        """
        Add a flagged item to the queue.

        Args:
            node: The node that failed audit
            tree_id: ID of the tree
            check_result: The audit check result
            rubric: Information preservation rubric
            full_input_a: Full (untruncated) input A
            full_input_b: Full (untruncated) input B

        Returns:
            The created FlaggedItem
        """
        self._item_counter += 1
        item_id = f"flag_{self._item_counter}"

        # Determine priority based on node level and discrepancy
        if check_result.discrepancy_score >= 0.8:
            priority = ReviewPriority.CRITICAL
        elif check_result.discrepancy_score >= 0.5:
            priority = ReviewPriority.HIGH
        elif node.level >= 2:
            priority = ReviewPriority.HIGH
        else:
            priority = ReviewPriority.MEDIUM

        item = FlaggedItem(
            item_id=item_id,
            node_id=node.id,
            tree_id=tree_id,
            check_type=check_result.check_type,
            input_a=full_input_a or check_result.input_a,
            input_b=full_input_b or check_result.input_b,
            rubric=rubric,
            approx_discrepancy=check_result.discrepancy_score,
            approx_reasoning=check_result.reasoning,
            priority=priority,
            node_level=node.level
        )

        # Enforce max size (remove lowest priority items)
        if len(self._items) >= self.max_size:
            self._evict_lowest_priority()

        self._items[item_id] = item
        return item

    def _evict_lowest_priority(self) -> None:
        """Remove lowest priority item to make room."""
        if not self._items:
            return
        # Sort by priority (ascending) and remove first
        sorted_items = sorted(
            self._items.values(),
            key=lambda x: (x.priority.value, x.approx_discrepancy)
        )
        if sorted_items:
            del self._items[sorted_items[0].item_id]

    def get_batch(
        self,
        limit: int = 10,
        priority_min: ReviewPriority = ReviewPriority.LOW,
        unreviewed_only: bool = True
    ) -> List[FlaggedItem]:
        """
        Get a batch of items for review.

        Args:
            limit: Maximum items to return
            priority_min: Minimum priority level
            unreviewed_only: Only return unreviewed items

        Returns:
            List of FlaggedItems sorted by priority (highest first)
        """
        items = list(self._items.values())

        # Filter
        if unreviewed_only:
            items = [i for i in items if not i.reviewed]
        items = [i for i in items if i.priority.value >= priority_min.value]

        # Sort by priority (descending), then discrepancy (descending)
        items.sort(key=lambda x: (-x.priority.value, -x.approx_discrepancy))

        return items[:limit]

    def get_all(self) -> List[FlaggedItem]:
        """Get all items in the queue."""
        return list(self._items.values())

    @property
    def items(self) -> List[FlaggedItem]:
        """Property to access all items (alias for get_all)."""
        return self.get_all()

    def add_item(self, item: FlaggedItem) -> None:
        """Add a pre-constructed FlaggedItem to the queue."""
        if len(self._items) >= self.max_size:
            self._evict_lowest_priority()
        self._items[item.item_id] = item

    def get_by_id(self, item_id: str) -> Optional[FlaggedItem]:
        """Get a specific item by ID."""
        return self._items.get(item_id)

    def update_item(self, item: FlaggedItem) -> None:
        """Update an item in the queue."""
        if item.item_id in self._items:
            self._items[item.item_id] = item

    def import_results(self, items: List[FlaggedItem]) -> int:
        """
        Import review results back into the queue.

        Args:
            items: List of reviewed FlaggedItems

        Returns:
            Number of items updated
        """
        updated = 0
        for item in items:
            if item.item_id in self._items:
                self._items[item.item_id] = item
                updated += 1
        return updated

    def export_to_json(self, filepath: str) -> None:
        """Export queue to JSON file for external processing."""
        data = {
            "exported_at": datetime.now().isoformat(),
            "item_count": len(self._items),
            "items": [item.to_dict() for item in self._items.values()]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def import_from_json(self, filepath: str) -> int:
        """Import items from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        items = [FlaggedItem.from_dict(d) for d in data.get("items", [])]
        return self.import_results(items)

    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        items = list(self._items.values())
        reviewed = [i for i in items if i.reviewed]
        approved = [i for i in reviewed if i.review_result]

        return {
            "total_items": len(items),
            "pending_review": len(items) - len(reviewed),
            "reviewed": len(reviewed),
            "approved": len(approved),
            "rejected": len(reviewed) - len(approved),
            "by_priority": {
                p.name: len([i for i in items if i.priority == p])
                for p in ReviewPriority
            },
            "avg_discrepancy": (
                sum(i.approx_discrepancy for i in items) / len(items)
                if items else 0.0
            )
        }

    def clear(self) -> None:
        """Clear all items from the queue."""
        self._items.clear()

    def get_reviewed_items(self) -> List[FlaggedItem]:
        """Get all reviewed items (for training data extraction)."""
        return [item for item in self._items.values() if item.reviewed]

    def get_unreviewed_items(self) -> List[FlaggedItem]:
        """Get all unreviewed items."""
        return [item for item in self._items.values() if not item.reviewed]

    def get_items_with_corrections(self) -> List[FlaggedItem]:
        """Get reviewed items that have corrected summaries."""
        return [
            item for item in self._items.values()
            if item.reviewed and item.corrected_summary
        ]

    def auto_review_with_oracle_func(
        self,
        oracle_func_engine: 'OracleFuncReviewEngine',
        auto_apply: bool = True,
        priority_min: 'ReviewPriority' = None
    ) -> List['OracleFuncReviewResult']:
        """
        Review unreviewed items using the learned oracle function approximation.

        This method integrates the oracle function approximation system with the
        review queue, allowing automated review of flagged nodes.

        Args:
            oracle_func_engine: Configured OracleFuncReviewEngine instance
            auto_apply: If True, automatically apply high-confidence decisions
            priority_min: Minimum priority level to review (default: MEDIUM)

        Returns:
            List of OracleFuncReviewResult for each reviewed item

        Example:
            >>> from src.ops_engine.oracle_func_approximation import OracleFuncReviewEngine, OracleFuncConfig
            >>> queue = ReviewQueue()
            >>> # ... audit items and add to queue ...
            >>> engine = OracleFuncReviewEngine(config=OracleFuncConfig(), review_queue=queue)
            >>> engine.train()  # Train from existing reviewed items
            >>> results = queue.auto_review_with_oracle_func(engine)
        """
        if priority_min is None:
            priority_min = ReviewPriority.MEDIUM

        return oracle_func_engine.review_flagged_nodes(
            queue=self,
            auto_apply=auto_apply,
            priority_min=priority_min
        )

    def __len__(self) -> int:
        return len(self._items)


class OPSAuditor:
    """
    Auditor for OPS trees.

    Performs probabilistic verification of summarization quality by
    sampling nodes and checking for information preservation.

    Two types of checks:
    1. Sufficiency Check (leaves): Does summary capture rubric info from raw text?
    2. Merge Consistency (internal): Does parent preserve info from children?

    Example:
        >>> oracle = create_oracle_from_scorer(SimpleScorer(), threshold=0.3)
        >>> queue = ReviewQueue()
        >>> auditor = OPSAuditor(oracle, config=AuditConfig(sample_budget=5), review_queue=queue)
        >>> report = auditor.audit_tree(tree)
        >>> print(f"Passed: {report.passed}, Failures: {report.nodes_failed}")
        >>> # Get flagged items for human review
        >>> batch = queue.get_batch(limit=10)
    """

    def __init__(
        self,
        oracle: Union[OracleJudge, ScoringOracle],
        config: Optional[AuditConfig] = None,
        review_queue: Optional[ReviewQueue] = None,
        summarizer: Optional[Callable[[str], str]] = None
    ):
        """
        Initialize the auditor.

        Args:
            oracle: Oracle judge for comparing inputs. Accepts both legacy
                OracleJudge (returns tuple) and new ScoringOracle (returns OracleScore).
                ScoringOracle is automatically wrapped for internal use.
            config: Audit configuration
            review_queue: Optional queue for flagging failures for batch review
            summarizer: Optional summarizer function for idempotence/substitution checks.
                       Required if audit_idempotence or audit_substitution is True.
        """
        # Normalize oracle interface: detect ScoringOracle vs legacy OracleJudge
        # ScoringOracles have .score() method returning OracleScore
        # Legacy OracleJudges have __call__ returning (bool, float, str)
        def _has_call_method(obj) -> bool:
            """Check if object has __call__ defined (excluding object base)."""
            for cls in type(obj).__mro__:
                if cls is object:
                    continue
                if '__call__' in cls.__dict__:
                    return True
            return False

        if hasattr(oracle, 'score') and not _has_call_method(oracle):
            # Pure ScoringOracle (has score but no __call__), wrap to legacy interface
            threshold = config.discrepancy_threshold if config else 0.1
            # LegacyOracleAdapter threshold is similarity (1.0=good), convert from discrepancy
            self.oracle = LegacyOracleAdapter(oracle, threshold=1.0 - threshold)
        else:
            self.oracle = oracle
        self.config = config or AuditConfig()
        self.review_queue = review_queue
        self.summarizer = summarizer

        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)

    def audit_tree(self, tree: OPSTree) -> AuditReport:
        """
        Audit an OPS tree.

        Samples nodes and checks for information preservation. Failures
        are automatically flagged to the review queue if one is configured.

        Performs four types of checks (from paper Section 4.1):
        1. Sufficiency (C1): Does leaf summary preserve oracle info from raw text?
        2. Merge Consistency (C3 Case B): Does internal merge preserve oracle info?
        3. Idempotence (C2): Does re-summarizing a summary keep oracle unchanged?
        4. Substitution (C3 Case A): Do joint vs disjoint summary paths agree?

        Args:
            tree: The tree to audit

        Returns:
            AuditReport with results
        """
        tree_id = tree.root.id if tree.root else "unknown"
        rubric = tree.rubric

        all_nodes = list(tree.traverse_preorder())
        leaves = [n for n in all_nodes if n.is_leaf]
        internal = [n for n in all_nodes if not n.is_leaf]

        checks = []
        audited_ids: Set[str] = set()

        # Track violation counts for computing rates
        sufficiency_violations = 0
        sufficiency_samples = 0
        merge_violations = 0
        merge_samples = 0
        idempotence_violations = 0
        idempotence_samples = 0
        substitution_violations = 0
        substitution_samples = 0

        # Determine how to split budget between leaves and internal nodes
        leaf_budget = self.config.sample_budget // 2 if self.config.audit_internal else self.config.sample_budget
        internal_budget = self.config.sample_budget - leaf_budget

        # Audit leaves (sufficiency check - C1) - concurrent for better GPU utilization
        if self.config.audit_leaves and leaves:
            leaf_samples = self._sample_nodes(leaves, leaf_budget)
            leaf_results = self._batch_audit_nodes(leaf_samples, self._check_sufficiency, rubric)
            for result, full_a, full_b, node in leaf_results:
                checks.append(result)
                audited_ids.add(node.id)
                self._update_node_audit(node, result, tree_id, rubric, full_a, full_b)
                sufficiency_samples += 1
                if not result.passed:
                    sufficiency_violations += 1

        # Audit internal nodes (merge consistency check - C3 Case B) - concurrent
        if self.config.audit_internal and internal:
            internal_samples = self._sample_nodes(internal, internal_budget)
            internal_results = self._batch_audit_nodes(internal_samples, self._check_merge_consistency, rubric)
            for result, full_a, full_b, node in internal_results:
                checks.append(result)
                audited_ids.add(node.id)
                self._update_node_audit(node, result, tree_id, rubric, full_a, full_b)
                merge_samples += 1
                if not result.passed:
                    merge_violations += 1

        # Idempotence check (C2) - Re-summarize summaries and check oracle stability - concurrent
        if self.config.audit_idempotence and internal and self.summarizer is not None:
            idem_samples = self._sample_nodes(internal, self.config.idempotence_budget)

            def check_idempotence(node):
                return self._check_idempotence(node, rubric)

            audit_workers = self.config.get_concurrency().audit_max_workers
            with ThreadPoolExecutor(max_workers=audit_workers) as executor:
                idem_results = list(executor.map(check_idempotence, idem_samples))

            for result in idem_results:
                checks.append(result)
                idempotence_samples += 1
                if not result.passed:
                    idempotence_violations += 1

        # Substitution check (C3 Case A) - Check leaf boundary consistency - concurrent
        if self.config.audit_substitution and len(leaves) >= 2 and self.summarizer is not None:
            # Get adjacent leaf pairs
            adjacent_pairs = self._get_adjacent_leaf_pairs(leaves)
            if adjacent_pairs:
                sub_budget = min(self.config.substitution_budget, len(adjacent_pairs))
                sampled_pairs = random.sample(adjacent_pairs, sub_budget)

                def check_substitution(pair):
                    left_node, right_node = pair
                    return self._check_substitution(left_node, right_node, rubric)

                sub_workers = self.config.get_concurrency().audit_max_workers
                with ThreadPoolExecutor(max_workers=sub_workers) as executor:
                    sub_results = list(executor.map(check_substitution, sampled_pairs))

                for result in sub_results:
                    checks.append(result)
                    substitution_samples += 1
                    if not result.passed:
                        substitution_violations += 1

        # Compile report
        passed = sum(1 for c in checks if c.passed)
        failed = len(checks) - passed
        failed_ids = [c.node_id for c in checks if not c.passed]

        return AuditReport(
            tree_id=tree_id,
            total_nodes=len(all_nodes),
            nodes_audited=len(checks),
            nodes_passed=passed,
            nodes_failed=failed,
            failure_rate=failed / len(checks) if checks else 0.0,
            checks=checks,
            failed_node_ids=failed_ids,
            sufficiency_violations=sufficiency_violations,
            merge_violations=merge_violations,
            idempotence_violations=idempotence_violations,
            substitution_violations=substitution_violations,
            sufficiency_samples=sufficiency_samples,
            merge_samples=merge_samples,
            idempotence_samples=idempotence_samples,
            substitution_samples=substitution_samples
        )

    def _batch_audit_nodes(
        self,
        nodes: List[OPSNode],
        check_fn: Callable,
        rubric: str,
        max_workers: Optional[int] = None
    ) -> List[Tuple["AuditCheckResult", str, str, OPSNode]]:
        """
        Run audit checks on nodes concurrently for better GPU utilization.

        Args:
            nodes: Nodes to audit
            check_fn: Check function (e.g., _check_sufficiency or _check_merge_consistency)
            rubric: Rubric for the oracle
            max_workers: Maximum concurrent workers (uses config default if None)

        Returns:
            List of (result, full_a, full_b, node) tuples for nodes that were audited
        """
        results = []

        def check_node(node):
            if self._should_audit(node):
                result, full_a, full_b = check_fn(node, rubric)
                return (result, full_a, full_b, node)
            return None

        # Use concurrency config if max_workers not explicitly set
        if max_workers is None:
            max_workers = self.config.get_concurrency().audit_max_workers

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(check_node, node) for node in nodes]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

        return results

    def _sample_nodes(self, nodes: List[OPSNode], budget: int) -> List[OPSNode]:
        """
        Sample nodes according to the configured strategy.

        Args:
            nodes: Nodes to sample from
            budget: Maximum nodes to sample

        Returns:
            List of sampled nodes
        """
        if not nodes:
            return []

        budget = min(budget, len(nodes))

        if self.config.sampling_strategy == SamplingStrategy.RANDOM:
            return random.sample(nodes, budget)

        elif self.config.sampling_strategy == SamplingStrategy.LEVEL_WEIGHTED:
            # Weight nodes by level (higher levels get more weight)
            max_level = max(n.level for n in nodes)
            weights = [(n.level + 1) / (max_level + 1) for n in nodes]

            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]

            # Sample without replacement using weights
            sampled = []
            available = list(zip(nodes, weights))

            for _ in range(budget):
                if not available:
                    break
                nodes_only = [n for n, _ in available]
                weights_only = [w for _, w in available]

                # Renormalize
                total = sum(weights_only)
                if total == 0:
                    break
                weights_only = [w / total for w in weights_only]

                chosen_idx = random.choices(range(len(nodes_only)), weights=weights_only, k=1)[0]
                sampled.append(nodes_only[chosen_idx])
                available.pop(chosen_idx)

            return sampled

        else:
            # Default to random
            return random.sample(nodes, budget)

    def _should_audit(self, node: OPSNode) -> bool:
        """
        Determine if a node should be audited based on probability.

        Args:
            node: Node to consider

        Returns:
            True if should audit
        """
        if self.config.sampling_probability >= 1.0:
            return True
        return random.random() < self.config.sampling_probability

    def _check_sufficiency(
        self, node: OPSNode, rubric: str
    ) -> Tuple[AuditCheckResult, str, str]:
        """
        Check if a leaf node's summary is sufficient.

        Compares raw text to summary.

        Args:
            node: Leaf node to check
            rubric: Information preservation rubric

        Returns:
            Tuple of (AuditCheckResult, full_input_a, full_input_b)
        """
        if not node.is_leaf:
            logger.warning(f"Sufficiency check called on non-leaf node {node.id}")

        input_a = node.raw_text_span or ""
        input_b = node.summary

        is_congruent, score, reasoning = self.oracle(input_a, input_b, rubric)
        passed = is_congruent and score <= self.config.discrepancy_threshold

        result = AuditCheckResult(
            node_id=node.id,
            check_type="sufficiency",
            passed=passed,
            discrepancy_score=score,
            reasoning=reasoning,
            input_a=input_a[:200],  # Truncate for report
            input_b=input_b[:200]
        )
        return result, input_a, input_b

    def _check_merge_consistency(
        self, node: OPSNode, rubric: str
    ) -> Tuple[AuditCheckResult, str, str]:
        """
        Check if an internal node's summary is consistent with its children.

        Compares concatenated child summaries to parent summary.

        Args:
            node: Internal node to check
            rubric: Information preservation rubric

        Returns:
            Tuple of (AuditCheckResult, full_input_a, full_input_b)
        """
        if node.is_leaf:
            logger.warning(f"Merge check called on leaf node {node.id}")

        # Concatenate child summaries
        child_summaries = []
        if node.left_child:
            child_summaries.append(node.left_child.summary)
        if node.right_child:
            child_summaries.append(node.right_child.summary)

        input_a = "\n\n".join(child_summaries)
        input_b = node.summary

        is_congruent, score, reasoning = self.oracle(input_a, input_b, rubric)
        passed = is_congruent and score <= self.config.discrepancy_threshold

        result = AuditCheckResult(
            node_id=node.id,
            check_type="merge_consistency",
            passed=passed,
            discrepancy_score=score,
            reasoning=reasoning,
            input_a=input_a[:200],
            input_b=input_b[:200]
        )
        return result, input_a, input_b

    def _check_idempotence(
        self, node: OPSNode, rubric: str
    ) -> AuditCheckResult:
        """
        Check if re-summarizing a summary preserves the oracle (Condition C2).

        From paper Section 4.1 (Sampling Stability):
        p_idem := (1/n_s) * sum I{d_Y(f*(g(z_k)), f*(z_k)) > τ}

        This samples summaries that have already passed through g and checks
        if re-summarization alters the oracle.

        Args:
            node: Internal node whose summary to re-summarize
            rubric: Information preservation rubric

        Returns:
            AuditCheckResult for the idempotence check
        """
        if self.summarizer is None:
            logger.warning("Idempotence check requires summarizer to be configured")
            return AuditCheckResult(
                node_id=node.id,
                check_type="idempotence",
                passed=True,
                discrepancy_score=0.0,
                reasoning="Skipped: no summarizer configured"
            )

        # The original summary s
        original_summary = node.summary

        # Re-summarize: g(s, rubric)
        try:
            re_summarized = self.summarizer(original_summary, rubric)
        except Exception as e:
            logger.error(f"Summarizer failed during idempotence check: {e}")
            return AuditCheckResult(
                node_id=node.id,
                check_type="idempotence",
                passed=False,
                discrepancy_score=1.0,
                reasoning=f"Summarizer error: {e}"
            )

        # Compare f*(s) vs f*(g(s))
        is_congruent, score, reasoning = self.oracle(original_summary, re_summarized, rubric)
        passed = is_congruent and score <= self.config.discrepancy_threshold

        return AuditCheckResult(
            node_id=node.id,
            check_type="idempotence",
            passed=passed,
            discrepancy_score=score,
            reasoning=f"Idempotence: {reasoning}",
            input_a=original_summary[:200],
            input_b=re_summarized[:200]
        )

    def _check_substitution(
        self, left_node: OPSNode, right_node: OPSNode, rubric: str
    ) -> AuditCheckResult:
        """
        Check leaf boundary substitution consistency (Condition C3 Case A).

        From paper Section 4.1 (Sampling Merge Consistency - Case A):
        When u, v are adjacent raw blocks, u ⊕ v fits in context.
        Compare the joint and disjoint summaries:
        I_bound := I{d_Y(f*(g(u⊕v)), f*(g(g(u)⊕g(v)))) > τ}

        This tests whether summarizing the joint raw span gives the same
        oracle result as first summarizing each part then merging.

        Args:
            left_node: Left leaf node in the adjacent pair
            right_node: Right leaf node in the adjacent pair
            rubric: Information preservation rubric

        Returns:
            AuditCheckResult for the substitution check
        """
        if self.summarizer is None:
            logger.warning("Substitution check requires summarizer to be configured")
            return AuditCheckResult(
                node_id=f"{left_node.id}+{right_node.id}",
                check_type="substitution",
                passed=True,
                discrepancy_score=0.0,
                reasoning="Skipped: no summarizer configured"
            )

        # Get raw text spans
        raw_left = left_node.raw_text_span or ""
        raw_right = right_node.raw_text_span or ""

        # Joint path: g(u ⊕ v, rubric) - summarize the concatenated raw text directly
        joint_raw = raw_left + "\n\n" + raw_right
        try:
            joint_summary = self.summarizer(joint_raw, rubric)
        except Exception as e:
            logger.error(f"Summarizer failed on joint text: {e}")
            return AuditCheckResult(
                node_id=f"{left_node.id}+{right_node.id}",
                check_type="substitution",
                passed=False,
                discrepancy_score=1.0,
                reasoning=f"Joint summarizer error: {e}"
            )

        # Disjoint path: g(g(u) ⊕ g(v)) - summarize parts first, then concatenate and summarize again
        # Use existing summaries if available, otherwise generate them
        left_summary = left_node.summary if left_node.summary else self.summarizer(raw_left, rubric)
        right_summary = right_node.summary if right_node.summary else self.summarizer(raw_right, rubric)

        # Concatenate child summaries and re-summarize
        concat_summaries = left_summary + "\n\n" + right_summary
        try:
            disjoint_summary = self.summarizer(concat_summaries, rubric)
        except Exception as e:
            logger.error(f"Summarizer failed on disjoint path: {e}")
            return AuditCheckResult(
                node_id=f"{left_node.id}+{right_node.id}",
                check_type="substitution",
                passed=False,
                discrepancy_score=1.0,
                reasoning=f"Disjoint summarizer error: {e}"
            )

        # Compare f*(joint_summary) vs f*(disjoint_summary)
        is_congruent, score, reasoning = self.oracle(joint_summary, disjoint_summary, rubric)
        passed = is_congruent and score <= self.config.discrepancy_threshold

        return AuditCheckResult(
            node_id=f"{left_node.id}+{right_node.id}",
            check_type="substitution",
            passed=passed,
            discrepancy_score=score,
            reasoning=f"Substitution (joint vs disjoint): {reasoning}",
            input_a=joint_summary[:200],
            input_b=disjoint_summary[:200]
        )

    def _get_adjacent_leaf_pairs(
        self, leaves: List[OPSNode]
    ) -> List[Tuple[OPSNode, OPSNode]]:
        """
        Get pairs of adjacent leaf nodes for substitution checks.

        Adjacent leaves are those that appear consecutively in the
        document's original text order.

        Args:
            leaves: List of leaf nodes from the tree

        Returns:
            List of (left_node, right_node) tuples for adjacent pairs
        """
        if len(leaves) < 2:
            return []

        # Sort leaves by their position in the document
        # Assuming leaves have some ordering info (id, span_start, etc.)
        # For now, use the order they appear in the list (left-to-right traversal)
        # In a proper implementation, leaves should be sorted by document position

        # Try to sort by span_start if available, otherwise use list order
        try:
            sorted_leaves = sorted(
                leaves,
                key=lambda n: getattr(n, 'span_start', 0) or 0
            )
        except (AttributeError, TypeError):
            sorted_leaves = leaves

        pairs = []
        for i in range(len(sorted_leaves) - 1):
            pairs.append((sorted_leaves[i], sorted_leaves[i + 1]))

        return pairs

    def _update_node_audit(
        self,
        node: OPSNode,
        result: AuditCheckResult,
        tree_id: str,
        rubric: str,
        full_input_a: str,
        full_input_b: str
    ) -> None:
        """
        Update node's audit status and flag failures to review queue.

        Args:
            node: The node being audited
            result: The audit check result
            tree_id: ID of the tree
            rubric: Information preservation rubric
            full_input_a: Full (untruncated) input A
            full_input_b: Full (untruncated) input B
        """
        if result.passed:
            node.set_audit_passed(result.discrepancy_score, result.reasoning)
        else:
            node.set_audit_failed(result.discrepancy_score, result.reasoning)

            # Flag to review queue if configured
            if self.review_queue is not None:
                self.review_queue.add(
                    node=node,
                    tree_id=tree_id,
                    check_result=result,
                    rubric=rubric,
                    full_input_a=full_input_a,
                    full_input_b=full_input_b
                )


def audit_tree(
    tree: OPSTree,
    oracle: Optional[Union[OracleJudge, Callable]] = None,
    scorer: Optional[ScoringOracle] = None,
    sample_budget: int = 10,
    threshold: float = 0.1
) -> AuditReport:
    """
    Convenience function to audit an OPS tree.

    Args:
        tree: Tree to audit
        oracle: Legacy oracle callable (deprecated, use scorer instead)
        scorer: ScoringOracle instance (preferred, e.g., SimpleScorer)
        sample_budget: Number of nodes to sample
        threshold: Discrepancy threshold

    Returns:
        AuditReport

    Example (preferred - using new ScoringOracle API):
        from src.ops_engine.scoring import SimpleScorer
        report = audit_tree(tree, scorer=SimpleScorer(), threshold=0.1)

    Example (legacy - still supported but deprecated):
        report = audit_tree(tree, oracle=my_oracle, threshold=0.1)
    """
    if oracle is None:
        if scorer is not None:
            # Use the new ScoringOracle API with adapter
            oracle = create_oracle_from_scorer(scorer, threshold=threshold)
        else:
            # Default: use SimpleScorer with adapter (no deprecation warning)
            oracle = create_oracle_from_scorer(SimpleScorer(), threshold=threshold)

    config = AuditConfig(
        sample_budget=sample_budget,
        discrepancy_threshold=threshold
    )

    auditor = OPSAuditor(oracle, config)
    return auditor.audit_tree(tree)


def get_human_review_queue(report: AuditReport) -> List[str]:
    """
    Get list of node IDs that need human review.

    Args:
        report: Audit report

    Returns:
        List of failed node IDs
    """
    return report.failed_node_ids


def compute_violation_bound(
    report: AuditReport,
    num_leaves: int,
    num_merges: Optional[int] = None,
    num_rounds: int = 1
) -> float:
    """
    Compute the global violation bound from the paper (Equation 1).

    From paper Section 4.1 (Scaling with tree size):
    Pr[root violation] ≤ N * p_suff + M * p_assoc + (R-1) * p_idem

    where:
    - N = number of leaves
    - M = number of merges (N-1 for binary tree)
    - R = number of re-summarization rounds
    - p_suff = sufficiency violation rate
    - p_assoc = combined merge-consistency rate (weighted avg of p_bound and p_merge)
    - p_idem = idempotence violation rate

    This provides a transparent union bound on the probability that the
    root summary deviates from the oracle.

    Args:
        report: Audit report containing violation rates
        num_leaves: Number of leaves in the tree (N)
        num_merges: Number of internal merges (M), defaults to N-1
        num_rounds: Number of re-summarization rounds (R)

    Returns:
        Upper bound on root violation probability (capped at 1.0)
    """
    if num_merges is None:
        num_merges = max(0, num_leaves - 1)

    # Get violation rates from report
    p_suff = report.sufficiency_rate
    p_assoc = report.assoc_rate
    p_idem = report.idempotence_rate

    # Compute bound: N * p_suff + M * p_assoc + (R-1) * p_idem
    bound = (
        num_leaves * p_suff +
        num_merges * p_assoc +
        max(0, num_rounds - 1) * p_idem
    )

    return min(bound, 1.0)


def compute_expected_distortion(
    report: AuditReport,
    num_leaves: int,
    num_merges: Optional[int] = None,
    num_rounds: int = 1
) -> float:
    """
    Compute the expected task-space distortion bound.

    From paper Appendix D (Equation 2):
    Δ_1 ≤ N * p_suff + M * p_assoc
    Δ_R ≤ N * p_suff + M * p_assoc + (R-1) * p_idem  (for R ≥ 2)

    Since d_Y ∈ [0,1], we have Δ_R = E[d_Y(·,·)] ≤ Pr(ε_R), so this
    bound also applies to expected distortion.

    Args:
        report: Audit report containing violation rates
        num_leaves: Number of leaves in the tree (N)
        num_merges: Number of internal merges (M), defaults to N-1
        num_rounds: Number of re-summarization rounds (R)

    Returns:
        Upper bound on expected distortion (capped at 1.0)
    """
    # Same computation as violation bound for bounded metrics
    return compute_violation_bound(report, num_leaves, num_merges, num_rounds)


def get_audit_statistics(report: AuditReport) -> Dict[str, Any]:
    """
    Get a summary of audit statistics for reporting.

    Returns a dictionary with all violation rates and sample counts,
    suitable for logging or display.

    Args:
        report: Audit report

    Returns:
        Dictionary of audit statistics
    """
    return {
        "tree_id": report.tree_id,
        "total_nodes": report.total_nodes,
        "nodes_audited": report.nodes_audited,
        "overall_passed": report.passed,
        "failure_rate": report.failure_rate,
        "violation_rates": {
            "p_suff": report.sufficiency_rate,
            "p_merge": report.merge_rate,
            "p_idem": report.idempotence_rate,
            "p_bound": report.substitution_rate,
            "p_assoc": report.assoc_rate,
        },
        "sample_counts": {
            "sufficiency": report.sufficiency_samples,
            "merge": report.merge_samples,
            "idempotence": report.idempotence_samples,
            "substitution": report.substitution_samples,
        },
        "violations": {
            "sufficiency": report.sufficiency_violations,
            "merge": report.merge_violations,
            "idempotence": report.idempotence_violations,
            "substitution": report.substitution_violations,
        }
    }
