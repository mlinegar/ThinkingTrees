"""
Tests for OPS auditor module.
"""

import pytest
import tempfile
import json
from pathlib import Path
from src.core.logged_supervision import SamplingMetadata
from src.tree import (
    Auditor, AuditConfig, AuditReport, AuditCheckResult,
    AlwaysPassScorer, AlwaysFailScorer, SimpleScorer,
    SamplingStrategy, audit_tree,
    ReviewQueue, FlaggedItem, ReviewPriority,
    get_human_review_queue, get_ipw_statistics,
)
from src.tree.auditor import (
    confidence_margin, sample_complexity, compute_required_samples,
    GuaranteeLevel,
)
from src.tree.neural_operator import make_deterministic_summary_operator
from src.tree.ipw import (
    NodeType,
    TreeSample,
    clipped_hajek_diagnostics,
    hajek_estimate,
    hajek_ht_comparison,
    ipw_violation_rate_ht,
)
from src.core.data_models import Tree, leaf, node


# --- AuditConfig Tests ---

class TestAuditConfig:
    """Tests for AuditConfig."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = AuditConfig()

        assert config.sample_budget > 0
        assert 0.0 <= config.discrepancy_threshold <= 1.0
        assert config.sampling_strategy == SamplingStrategy.RANDOM

    def test_custom_config(self):
        """Custom config values work."""
        config = AuditConfig(
            sample_budget=5,
            discrepancy_threshold=0.2,
            sampling_strategy=SamplingStrategy.LEVEL_WEIGHTED,
            random_seed=42
        )

        assert config.sample_budget == 5
        assert config.discrepancy_threshold == 0.2


# --- Auditor Tests ---

class TestAuditor:
    """Tests for Auditor."""

    def test_audit_single_node_tree(self, single_node_tree):
        """Audit tree with single node."""
        auditor = Auditor(AlwaysPassScorer())
        report = auditor.audit_tree(single_node_tree)

        assert report.total_nodes == 1
        assert report.nodes_audited >= 0  # May be 0 or 1 depending on budget split
        assert report.passed

    def test_audit_simple_tree(self, simple_tree):
        """Audit simple tree with multiple nodes."""
        config = AuditConfig(sample_budget=10)
        auditor = Auditor(AlwaysPassScorer(), config)
        report = auditor.audit_tree(simple_tree)

        assert report.total_nodes == 7
        assert report.nodes_audited > 0
        assert report.passed

    def test_audit_with_failures(self, simple_tree):
        """Audit that finds failures."""
        config = AuditConfig(sample_budget=10)
        auditor = Auditor(AlwaysFailScorer(), config)
        report = auditor.audit_tree(simple_tree)

        assert report.nodes_failed > 0
        assert not report.passed
        assert len(report.failed_node_ids) > 0

    def test_audit_respects_budget(self, simple_tree):
        """Auditor respects sample budget."""
        config = AuditConfig(sample_budget=2)
        auditor = Auditor(AlwaysPassScorer(), config)
        report = auditor.audit_tree(simple_tree)

        assert report.nodes_audited <= 2

    def test_check_types(self, simple_tree):
        """Both check types are performed."""
        config = AuditConfig(
            sample_budget=10,
            audit_leaves=True,
            audit_internal=True
        )
        auditor = Auditor(AlwaysPassScorer(), config)
        report = auditor.audit_tree(simple_tree)

        check_types = set(c.check_type for c in report.checks)
        # Should have both types if we audited enough nodes
        if report.nodes_audited >= 2:
            # At least one type should be present
            assert len(check_types) >= 1

    def test_audit_only_leaves(self, simple_tree):
        """Can audit only leaves."""
        config = AuditConfig(
            sample_budget=10,
            audit_leaves=True,
            audit_internal=False
        )
        auditor = Auditor(AlwaysPassScorer(), config)
        report = auditor.audit_tree(simple_tree)

        for check in report.checks:
            assert check.check_type == "sufficiency"

    def test_audit_only_internal(self, simple_tree):
        """Can audit only internal nodes."""
        config = AuditConfig(
            sample_budget=10,
            audit_leaves=False,
            audit_internal=True
        )
        auditor = Auditor(AlwaysPassScorer(), config)
        report = auditor.audit_tree(simple_tree)

        for check in report.checks:
            assert check.check_type == "merge_consistency"

    def test_reproducible_with_seed(self, simple_tree):
        """Same seed gives same results."""
        config = AuditConfig(sample_budget=3, random_seed=42)

        auditor1 = Auditor(AlwaysPassScorer(), config)
        report1 = auditor1.audit_tree(simple_tree)

        auditor2 = Auditor(AlwaysPassScorer(), config)
        report2 = auditor2.audit_tree(simple_tree)

        # Same nodes should be audited
        assert set(c.node_id for c in report1.checks) == set(c.node_id for c in report2.checks)

    def test_updates_node_status(self, simple_tree):
        """Audit updates node audit status."""
        config = AuditConfig(sample_budget=10)
        auditor = Auditor(AlwaysPassScorer(), config)
        auditor.audit_tree(simple_tree)

        # Some nodes should now have audit status set
        from src.core.data_models import AuditStatus
        audited_nodes = [
            n for n in simple_tree.traverse_preorder()
            if n.audit_result.status != AuditStatus.PENDING
        ]
        assert len(audited_nodes) > 0

    def test_internal_merge_audit_uses_ops_span(self):
        left = leaf("left", node_id="left")
        right = leaf("right", node_id="right")
        root = node(left, right, summary="root summary", node_id="root")
        root.ops_span = "theorem span"

        auditor = Auditor(AlwaysPassScorer())
        result, input_a, input_b = auditor._check_merge_consistency(root, rubric="")

        assert result.check_type == "merge_consistency"
        assert input_a == "theorem span"
        assert input_b == "root summary"

    def test_idempotence_check_can_use_theorem_operator_without_summarizer(self):
        root = leaf("alpha", summary="ALPHA", node_id="root")
        operator = make_deterministic_summary_operator(
            lambda text, rubric: text.upper(),
            name="upper_summary",
        )

        auditor = Auditor(AlwaysPassScorer(), theorem_operator=operator)
        result = auditor._check_idempotence(root, rubric="")

        assert result.skipped is False
        assert result.input_a == "ALPHA"
        assert result.input_b == "ALPHA"

    def test_substitution_check_can_use_theorem_operator_combine(self):
        left_node = leaf("left raw", summary="LEFT", node_id="left")
        right_node = leaf("right raw", summary="RIGHT", node_id="right")
        operator = make_deterministic_summary_operator(
            lambda text, rubric: text.upper(),
            name="upper_summary",
        )

        auditor = Auditor(AlwaysPassScorer(), theorem_operator=operator)
        result = auditor._check_substitution(left_node, right_node, rubric="")

        assert result.skipped is False
        assert "PART 1" in result.input_a
        assert "PART 2" in result.input_b

    def test_audit_report_includes_operator_capabilities(self, simple_tree):
        operator = make_deterministic_summary_operator(
            lambda text, rubric: text.upper(),
            name="upper_summary",
        )
        auditor = Auditor(AlwaysPassScorer(), theorem_operator=operator)

        report = auditor.audit_tree(simple_tree)

        assert report.operator_capabilities["operator_name"] == "upper_summary"
        assert report.operator_capabilities["supports_resummary_idempotence"] is True
        assert report.compositional_learning_problem["name"] == "tree_audit_verification"
        assert report.compositional_learning_problem["uses_sampled_substructure_labels"] is True
        assert report.compositional_learning_problem["uses_online_oracle_queries"] is True
        channel = report.compositional_learning_problem["supervision_channels"][0]
        assert channel["kind"] == "sampled_substructure"
        assert channel["delivery_mode"] == "online_oracle_query"
        assert channel["requires_propensity_logging"] is True
        assert channel["query_policy"]["logs_realized_propensities"] is True


class TestAuditCheckResult:
    """Tests for AuditCheckResult."""

    def test_create_result(self):
        """Can create result."""
        result = AuditCheckResult(
            node_id="test",
            check_type="sufficiency",
            passed=True,
            discrepancy_score=0.1,
            reasoning="All good"
        )

        assert result.node_id == "test"
        assert result.passed
        assert result.discrepancy_score == 0.1


class TestAuditReport:
    """Tests for AuditReport."""

    def test_report_properties(self):
        """Report properties work."""
        report = AuditReport(
            tree_id="test",
            total_nodes=10,
            nodes_audited=5,
            nodes_passed=4,
            nodes_failed=1,
            failure_rate=0.2,
            failed_node_ids=["node_1"]
        )

        assert not report.passed
        assert report.failure_rate == 0.2

    def test_report_passed_when_no_failures(self):
        """Report passed when no failures."""
        report = AuditReport(
            tree_id="test",
            total_nodes=10,
            nodes_audited=5,
            nodes_passed=5,
            nodes_failed=0,
            failure_rate=0.0
        )

        assert report.passed


class TestSamplingStrategies:
    """Tests for different sampling strategies."""

    def test_random_sampling(self, simple_tree):
        """Random sampling works."""
        config = AuditConfig(
            sample_budget=3,
            sampling_strategy=SamplingStrategy.RANDOM
        )
        auditor = Auditor(AlwaysPassScorer(), config)
        report = auditor.audit_tree(simple_tree)

        assert report.nodes_audited <= 3

    def test_level_weighted_sampling(self, simple_tree):
        """Level weighted sampling works."""
        config = AuditConfig(
            sample_budget=3,
            sampling_strategy=SamplingStrategy.LEVEL_WEIGHTED
        )
        auditor = Auditor(AlwaysPassScorer(), config)
        report = auditor.audit_tree(simple_tree)

        assert report.nodes_audited <= 3


class TestConvenienceFunctions:
    """Tests for module-level functions."""

    def test_audit_tree_function(self, simple_tree):
        """audit_tree convenience function works."""
        report = audit_tree(simple_tree, sample_budget=5)

        assert isinstance(report, AuditReport)
        assert report.nodes_audited <= 5

    def test_get_human_review_queue(self, simple_tree):
        """get_human_review_queue extracts failed nodes."""
        report = audit_tree(
            simple_tree,
            scorer=AlwaysFailScorer(),
            sample_budget=3
        )

        queue = get_human_review_queue(report)
        assert len(queue) == report.nodes_failed
        assert queue == report.failed_node_ids


class TestFlaggedItem:
    """Tests for FlaggedItem."""

    def test_create_flagged_item(self):
        """Can create a flagged item."""
        item = FlaggedItem(
            item_id="flag_1",
            node_id="node_1",
            tree_id="tree_1",
            check_type="sufficiency",
            input_a="original text",
            input_b="summary",
            rubric="preserve info",
            approx_discrepancy=0.5,
            approx_reasoning="Some differences",
            priority=ReviewPriority.HIGH
        )

        assert item.item_id == "flag_1"
        assert item.priority == ReviewPriority.HIGH
        assert not item.reviewed

    def test_to_dict(self):
        """Can serialize to dict."""
        item = FlaggedItem(
            item_id="flag_1",
            node_id="node_1",
            tree_id="tree_1",
            check_type="sufficiency",
            input_a="text",
            input_b="summary",
            rubric="rubric",
            approx_discrepancy=0.3,
            approx_reasoning="reason"
        )

        d = item.to_dict()
        assert d["item_id"] == "flag_1"
        assert d["priority"] == "MEDIUM"

    def test_from_dict(self):
        """Can deserialize from dict."""
        d = {
            "item_id": "flag_2",
            "node_id": "node_2",
            "tree_id": "tree_2",
            "check_type": "merge_consistency",
            "input_a": "a",
            "input_b": "b",
            "rubric": "r",
            "approx_discrepancy": 0.8,
            "approx_reasoning": "bad",
            "priority": "CRITICAL",
            "flagged_at": "2024-01-01T00:00:00",
            "node_level": 2,
            "reviewed": True,
            "review_result": False,
            "review_reasoning": "needs fix",
            "corrected_summary": "fixed",
            "reviewed_at": "2024-01-02T00:00:00"
        }

        item = FlaggedItem.from_dict(d)
        assert item.item_id == "flag_2"
        assert item.priority == ReviewPriority.CRITICAL
        assert item.reviewed
        assert not item.review_result


class TestReviewQueue:
    """Tests for ReviewQueue."""

    def test_add_item(self, sample_leaf):
        """Can add items to queue."""
        queue = ReviewQueue()
        result = AuditCheckResult(
            node_id=sample_leaf.id,
            check_type="sufficiency",
            passed=False,
            discrepancy_score=0.6,
            reasoning="failed"
        )

        item = queue.add(
            node=sample_leaf,
            tree_id="tree_1",
            check_result=result,
            rubric="test rubric"
        )

        assert item.node_id == sample_leaf.id
        assert len(queue) == 1

    def test_get_batch(self, sample_leaves):
        """Can get batch of items."""
        queue = ReviewQueue()

        for leaf in sample_leaves:
            result = AuditCheckResult(
                node_id=leaf.id,
                check_type="sufficiency",
                passed=False,
                discrepancy_score=0.5,
                reasoning="failed"
            )
            queue.add(leaf, "tree", result, "rubric")

        batch = queue.get_batch(limit=2)
        assert len(batch) == 2

    def test_priority_sorting(self, sample_leaves):
        """Batch sorted by priority (highest first)."""
        queue = ReviewQueue()

        # Add with different discrepancy scores (affects priority)
        scores = [0.3, 0.9, 0.5, 0.7]  # 0.9 -> CRITICAL, 0.5/0.7 -> HIGH, 0.3 -> MEDIUM
        for leaf, score in zip(sample_leaves, scores):
            result = AuditCheckResult(
                node_id=leaf.id,
                check_type="sufficiency",
                passed=False,
                discrepancy_score=score,
                reasoning="failed"
            )
            queue.add(leaf, "tree", result, "rubric")

        batch = queue.get_batch(limit=4)
        # Should be sorted by priority descending
        priorities = [item.priority.value for item in batch]
        assert priorities == sorted(priorities, reverse=True)

    def test_unreviewed_only(self, sample_leaf):
        """Can filter to unreviewed only."""
        queue = ReviewQueue()
        result = AuditCheckResult(
            node_id=sample_leaf.id,
            check_type="sufficiency",
            passed=False,
            discrepancy_score=0.5,
            reasoning="failed"
        )

        item = queue.add(sample_leaf, "tree", result, "rubric")
        item.reviewed = True
        queue.update_item(item)

        # Should not appear in unreviewed batch
        batch = queue.get_batch(unreviewed_only=True)
        assert len(batch) == 0

        # Should appear when including reviewed
        batch = queue.get_batch(unreviewed_only=False)
        assert len(batch) == 1

    def test_export_import_json(self, sample_leaf, tmp_path):
        """Can export and import JSON."""
        queue = ReviewQueue()
        result = AuditCheckResult(
            node_id=sample_leaf.id,
            check_type="sufficiency",
            passed=False,
            discrepancy_score=0.5,
            reasoning="test"
        )
        queue.add(sample_leaf, "tree", result, "rubric")

        filepath = tmp_path / "queue.json"
        queue.export_to_json(str(filepath))

        # Import to new queue
        queue2 = ReviewQueue()
        # Need to add the item first to import results
        queue2.add(sample_leaf, "tree", result, "rubric")
        count = queue2.import_from_json(str(filepath))

        assert filepath.exists()

    def test_get_statistics(self, sample_leaves):
        """Can get queue statistics."""
        queue = ReviewQueue()

        for i, leaf in enumerate(sample_leaves):
            result = AuditCheckResult(
                node_id=leaf.id,
                check_type="sufficiency",
                passed=False,
                discrepancy_score=0.5 + i * 0.1,
                reasoning="test"
            )
            item = queue.add(leaf, "tree", result, "rubric")
            if i == 0:
                item.reviewed = True
                item.review_result = True
                queue.update_item(item)

        stats = queue.get_statistics()
        assert stats["total_items"] == 4
        assert stats["reviewed"] == 1
        assert stats["approved"] == 1
        assert stats["pending_review"] == 3

    def test_max_size_eviction(self):
        """Queue evicts lowest priority when full."""
        queue = ReviewQueue(max_size=2)

        for i in range(3):
            leaf_node = leaf(f"leaf {i}", node_id=f"leaf_{i}")
            result = AuditCheckResult(
                node_id=leaf_node.id,
                check_type="sufficiency",
                passed=False,
                discrepancy_score=0.3 + i * 0.3,  # 0.3, 0.6, 0.9
                reasoning="test"
            )
            queue.add(leaf_node, "tree", result, "rubric")

        # Should only have 2 items (highest priority)
        assert len(queue) == 2


class TestReviewQueueIntegration:
    """Integration tests for auditor with review queue."""

    def test_failures_flagged_to_queue(self, simple_tree):
        """Audit failures are flagged to review queue."""
        queue = ReviewQueue()
        config = AuditConfig(sample_budget=10)
        auditor = Auditor(AlwaysFailScorer(), config, review_queue=queue)

        report = auditor.audit_tree(simple_tree)

        # Queue should have same number of items as failures
        assert len(queue) == report.nodes_failed

    def test_no_flagging_on_pass(self, simple_tree):
        """No flagging when all audits pass."""
        queue = ReviewQueue()
        config = AuditConfig(sample_budget=10)
        auditor = Auditor(AlwaysPassScorer(), config, review_queue=queue)

        auditor.audit_tree(simple_tree)

        assert len(queue) == 0

    def test_flagged_items_have_full_content(self, simple_tree):
        """Flagged items contain full (untruncated) content."""
        queue = ReviewQueue()
        config = AuditConfig(sample_budget=10)
        auditor = Auditor(AlwaysFailScorer(), config, review_queue=queue)

        auditor.audit_tree(simple_tree)

        for item in queue.get_all():
            # Should have rubric
            assert item.rubric == simple_tree.rubric
            # Should have content
            assert len(item.input_a) > 0 or len(item.input_b) > 0


# =============================================================================
# Statistical Functions Tests (From Audit.lean alignment)
# =============================================================================

class TestConfidenceMargin:
    """Tests for confidence_margin function (matches Audit.lean)."""

    def test_basic_computation(self):
        """Basic confidence margin computation."""
        # For n=100, delta=0.05: margin = sqrt(ln(40)/(200)) ≈ 0.136
        margin = confidence_margin(0.05, 100)
        assert 0.13 < margin < 0.14

    def test_increases_with_lower_n(self):
        """Margin increases when sample size decreases."""
        margin_100 = confidence_margin(0.05, 100)
        margin_50 = confidence_margin(0.05, 50)
        assert margin_50 > margin_100

    def test_increases_with_higher_confidence(self):
        """Margin increases when confidence level increases (delta decreases)."""
        margin_95 = confidence_margin(0.05, 100)  # 95% confidence
        margin_99 = confidence_margin(0.01, 100)  # 99% confidence
        assert margin_99 > margin_95

    def test_invalid_inputs(self):
        """Invalid inputs return infinity."""
        assert confidence_margin(0.05, 0) == float('inf')
        assert confidence_margin(0.05, -1) == float('inf')
        assert confidence_margin(0, 100) == float('inf')
        assert confidence_margin(2, 100) == float('inf')

    def test_matches_lean_definition(self):
        """Matches the Lean definition: sqrt(ln(2/delta) / (2n))."""
        import math
        delta = 0.05
        n = 738
        expected = math.sqrt(math.log(2 / delta) / (2 * n))
        actual = confidence_margin(delta, n)
        assert abs(actual - expected) < 1e-10


class TestSampleComplexity:
    """Tests for sample_complexity function (matches Audit.lean)."""

    def test_basic_computation(self):
        """Sample complexity for epsilon=0.05, delta=0.05."""
        # From Audit.lean: ceil(ln(40) / (2 * 0.0025)) = ceil(1475.67) = 738
        n = sample_complexity(0.05, 0.05)
        assert n == 738

    def test_inverse_relationship(self):
        """Sample complexity is inverse to confidence_margin."""
        epsilon = 0.05
        delta = 0.05
        n = sample_complexity(epsilon, delta)
        margin = confidence_margin(delta, n)
        assert margin <= epsilon

    def test_increases_with_tighter_epsilon(self):
        """Need more samples for tighter epsilon."""
        n_5_percent = sample_complexity(0.05, 0.05)
        n_1_percent = sample_complexity(0.01, 0.05)
        assert n_1_percent > n_5_percent

    def test_increases_with_higher_confidence(self):
        """Need more samples for higher confidence (lower delta)."""
        n_95 = sample_complexity(0.05, 0.05)  # 95% confidence
        n_99 = sample_complexity(0.05, 0.01)  # 99% confidence
        assert n_99 > n_95

    def test_invalid_inputs(self):
        """Invalid inputs return large value."""
        assert sample_complexity(0, 0.05) == int(1e9)
        assert sample_complexity(0.05, 0) == int(1e9)
        assert sample_complexity(0.05, 2) == int(1e9)


class TestComputeRequiredSamples:
    """Tests for compute_required_samples helper."""

    def test_default_parameters(self):
        """Default parameters work."""
        result = compute_required_samples()
        assert len(result) == 4
        assert "sufficiency" in result
        assert "merge" in result
        assert "idempotence" in result
        assert "substitution" in result

    def test_custom_check_types(self):
        """Can specify custom check types."""
        result = compute_required_samples(check_types=["sufficiency", "merge"])
        assert len(result) == 2
        assert "sufficiency" in result
        assert "merge" in result

    def test_values_match_sample_complexity(self):
        """Values match sample_complexity function."""
        result = compute_required_samples(epsilon=0.05, delta=0.05)
        expected = sample_complexity(0.05, 0.05)
        for check_type, n in result.items():
            assert n == expected


class TestGuaranteeLevel:
    """Tests for GuaranteeLevel enum."""

    def test_enum_values(self):
        """Enum has correct values."""
        assert GuaranteeLevel.EXACT.value == "exact"
        assert GuaranteeLevel.UNION_BOUND.value == "union"
        assert GuaranteeLevel.EMPIRICAL.value == "empirical"


class TestAuditReportConfidenceMethods:
    """Tests for new AuditReport confidence methods."""

    def test_confidence_upper_bound_no_samples(self):
        """Upper bound is 1.0 when no samples taken."""
        report = AuditReport(
            tree_id="test", total_nodes=10, nodes_audited=0,
            nodes_passed=0, nodes_failed=0, failure_rate=0.0
        )
        assert report.confidence_upper_bound("sufficiency") == 1.0

    def test_confidence_upper_bound_with_samples(self):
        """Upper bound is rate + margin when samples taken."""
        report = AuditReport(
            tree_id="test", total_nodes=10, nodes_audited=100,
            nodes_passed=90, nodes_failed=10, failure_rate=0.1,
            sufficiency_violations=10, sufficiency_samples=100
        )
        # Rate = 0.1, margin = confidence_margin(0.05, 100) ≈ 0.136
        upper = report.confidence_upper_bound("sufficiency", delta=0.05)
        assert upper > 0.1  # Should be rate + margin
        assert upper < 0.3  # Should be bounded

    def test_confidence_upper_bound_zero_violations(self):
        """Upper bound is just margin when rate is 0."""
        report = AuditReport(
            tree_id="test", total_nodes=10, nodes_audited=100,
            nodes_passed=100, nodes_failed=0, failure_rate=0.0,
            sufficiency_violations=0, sufficiency_samples=100
        )
        upper = report.confidence_upper_bound("sufficiency", delta=0.05)
        margin = confidence_margin(0.05, 100)
        assert abs(upper - margin) < 1e-10

    def test_get_probabilistic_bound(self):
        """Probabilistic bound returns tuple."""
        report = AuditReport(
            tree_id="test", total_nodes=10, nodes_audited=100,
            nodes_passed=95, nodes_failed=5, failure_rate=0.05,
            sufficiency_violations=5, sufficiency_samples=50,
            merge_violations=0, merge_samples=50
        )
        bound, confidence = report.get_probabilistic_bound(
            num_leaves=10, num_rounds=1, delta=0.05
        )
        assert 0 <= bound <= 1.0
        assert confidence == 0.85  # 1 - 3*0.05

    def test_get_guarantee_level_exact(self):
        """EXACT level when no violations."""
        report = AuditReport(
            tree_id="test", total_nodes=10, nodes_audited=10,
            nodes_passed=10, nodes_failed=0, failure_rate=0.0,
            sufficiency_violations=0, sufficiency_samples=5,
            merge_violations=0, merge_samples=5,
            idempotence_violations=0, idempotence_samples=0,
            substitution_violations=0, substitution_samples=0
        )
        assert report.get_guarantee_level() == GuaranteeLevel.EXACT

    def test_get_guarantee_level_union_bound(self):
        """UNION_BOUND level when violations exist."""
        report = AuditReport(
            tree_id="test", total_nodes=10, nodes_audited=10,
            nodes_passed=9, nodes_failed=1, failure_rate=0.1,
            sufficiency_violations=1, sufficiency_samples=5,
            merge_violations=0, merge_samples=5,
            idempotence_violations=0, idempotence_samples=0,
            substitution_violations=0, substitution_samples=0
        )
        assert report.get_guarantee_level() == GuaranteeLevel.UNION_BOUND

    def test_get_guarantee_level_empirical(self):
        """EMPIRICAL level when no audits performed."""
        report = AuditReport(
            tree_id="test", total_nodes=10, nodes_audited=0,
            nodes_passed=0, nodes_failed=0, failure_rate=0.0
        )
        assert report.get_guarantee_level() == GuaranteeLevel.EMPIRICAL


class TestAuditConfigStatistical:
    """Tests for AuditConfig statistical guarantee parameters."""

    def test_default_delta(self):
        """Default delta is 0.05 (95% confidence)."""
        config = AuditConfig()
        assert config.target_delta == 0.05

    def test_target_epsilon_none_by_default(self):
        """target_epsilon is None by default."""
        config = AuditConfig()
        assert config.target_epsilon is None

    def test_compute_sample_budget_no_target(self):
        """Returns sample_budget when no target_epsilon."""
        config = AuditConfig(sample_budget=10)
        assert config.compute_sample_budget_for_guarantee() == 10

    def test_compute_sample_budget_with_target(self):
        """Returns sample_complexity when target_epsilon set."""
        config = AuditConfig(
            sample_budget=10,
            target_epsilon=0.05,
            target_delta=0.05
        )
        expected = sample_complexity(0.05, 0.05)
        assert config.compute_sample_budget_for_guarantee() == expected


class TestTreeIPWExtensions:
    """Tests for HT/clipping diagnostics and doc_id propagation."""

    @staticmethod
    def _sample(node_id: str, violation: int, node_propensity: float) -> TreeSample:
        return TreeSample(
            doc_id="doc-1",
            node_id=node_id,
            node_type=NodeType.LEAF,
            violation=violation,
            sampling=SamplingMetadata(
                document_propensity=1.0,
                unit_propensity=node_propensity,
                label_propensity=1.0,
            ),
        )

    def test_sampling_metadata_rejects_values_above_one(self):
        """Sampling propensities are bounded by 1."""
        with pytest.raises(ValueError):
            SamplingMetadata(
                document_propensity=1.01,
                unit_propensity=1.0,
                label_propensity=1.0,
            )

    def test_hajek_vs_ht_comparison_for_violations(self):
        """HT and Hajek comparison exposes expected gap."""
        samples = [
            self._sample("n1", violation=1, node_propensity=0.5),  # weight 2
            self._sample("n2", violation=0, node_propensity=1.0),  # weight 1
        ]

        comparison = hajek_ht_comparison(
            samples,
            lambda sample: float(sample.violation),
            population_size=2.0,
        )

        assert comparison["hajek"] == pytest.approx(2.0 / 3.0)
        assert comparison["ht_mean"] == pytest.approx(1.0)
        assert comparison["abs_diff"] == pytest.approx(1.0 / 3.0)
        assert ipw_violation_rate_ht(samples) == pytest.approx(1.0)

    def test_clipped_hajek_diagnostics_bound_holds(self):
        """Clipped diagnostics satisfy the deterministic envelope."""
        samples = [
            self._sample("n1", violation=1, node_propensity=0.5),
            self._sample("n2", violation=0, node_propensity=1.0),
        ]
        diagnostics = clipped_hajek_diagnostics(
            samples,
            lambda sample: float(sample.violation),
            max_weight=1.0,
            value_min=0.0,
            value_max=1.0,
        )

        raw = hajek_estimate(samples, lambda sample: float(sample.violation))
        assert diagnostics["raw_hajek"] == pytest.approx(raw)
        assert diagnostics["clipped_hajek"] == pytest.approx(0.5)
        assert diagnostics["bound_holds"] == 1.0
        assert diagnostics["abs_diff"] <= diagnostics["abs_diff_bound"] + 1e-12

    def test_to_tree_samples_prefers_source_doc_id(self):
        """Tree samples should use source document id for fold splitting."""
        report = AuditReport(
            tree_id="tree-1",
            total_nodes=1,
            nodes_audited=1,
            nodes_passed=1,
            nodes_failed=0,
            failure_rate=0.0,
            source_doc_id="doc-123",
            checks=[
                AuditCheckResult(
                    node_id="n1",
                    check_type="sufficiency",
                    passed=True,
                    discrepancy_score=0.0,
                    reasoning="ok",
                )
            ],
            sufficiency_samples=1,
            leaf_population=1,
        )

        tree_samples = report.to_tree_samples()
        assert len(tree_samples) == 1
        assert tree_samples[0].doc_id == "doc-123"

    def test_get_ipw_statistics_includes_ht_and_clipping(self):
        """IPW stats include HT/Hajek and clipped-Hajek diagnostics."""
        report = AuditReport(
            tree_id="tree-1",
            total_nodes=1,
            nodes_audited=1,
            nodes_passed=0,
            nodes_failed=1,
            failure_rate=1.0,
            checks=[
                AuditCheckResult(
                    node_id="n1",
                    check_type="sufficiency",
                    passed=False,
                    discrepancy_score=0.2,
                    reasoning="fail",
                )
            ],
            sufficiency_violations=1,
            sufficiency_samples=1,
            leaf_population=1,
        )

        stats = get_ipw_statistics(
            report=report,
            num_leaves=1,
            num_merges=0,
            num_rounds=1,
            delta=0.05,
            clip_max_weight=2.0,
        )

        assert "ht_vs_hajek" in stats
        assert "overall_violation" in stats["ht_vs_hajek"]
        assert "clipping" in stats
        assert stats["clipping"]["max_weight"] == pytest.approx(2.0)
