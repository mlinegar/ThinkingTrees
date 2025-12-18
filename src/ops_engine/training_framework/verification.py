"""
OPS Law Verification at Tree Nodes.

This module implements verification of OPS (Oracle-Preserving Summarization) laws
at each node in the summarization tree. The verifier uses the OracleClassifier to
check that summaries preserve oracle-relevant information according to the laws:

- C1 (Sufficiency): oracle(summary) ≈ oracle(original)
- C2 (Idempotence): oracle(summarize(S)) ≈ oracle(S)
- C3A (Substitution): Equivalent summaries → same oracle
- C3B (Merge Consistency): oracle(merge) ≈ aggregate(oracle(children))
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
import uuid

from .core import (
    LabelSpace,
    Prediction,
    LawCheckResult,
    ViolationType,
    UnifiedTrainingExample,
    TrainingExampleLabel,
)
from .inference import OracleClassifier


@dataclass
class NodeVerificationResult:
    """Complete verification result for a single tree node."""
    node_id: str
    law_results: Dict[str, LawCheckResult]  # law_name -> result

    @property
    def all_passed(self) -> bool:
        """Whether all law checks passed."""
        return all(r.passed for r in self.law_results.values())

    @property
    def failed_laws(self) -> List[str]:
        """List of laws that failed."""
        return [law for law, result in self.law_results.items() if not result.passed]

    @property
    def total_discrepancy(self) -> float:
        """Sum of discrepancies across all checks."""
        return sum(r.discrepancy for r in self.law_results.values())

    def to_training_examples(
        self,
        original_content: str,
        summary: str,
        rubric: str,
    ) -> List[UnifiedTrainingExample]:
        """Convert all law check results to training examples."""
        examples = []
        for law_name, result in self.law_results.items():
            example_id = f"{self.node_id}_{law_name}_{uuid.uuid4().hex[:8]}"
            examples.append(result.to_training_example(
                original_content=original_content,
                summary=summary,
                rubric=rubric,
                example_id=example_id,
            ))
        return examples


class OracleNodeVerifier:
    """
    Verifies OPS law compliance at tree nodes.

    Uses an OracleClassifier to predict labels for original content and summaries,
    then compares predictions to check if OPS laws are satisfied.
    """

    def __init__(
        self,
        classifier: OracleClassifier,
        tolerance: float = 0.0,
        summarizer: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the verifier.

        Args:
            classifier: The oracle classifier to use for predictions
            tolerance: Allowed discrepancy before marking as violation (useful for ordinal)
            summarizer: Optional function to re-summarize for idempotence checks
        """
        self.classifier = classifier
        self.tolerance = tolerance
        self.summarizer = summarizer

    def _labels_equivalent(self, label_a: str, label_b: str) -> bool:
        """Check if two labels are equivalent within tolerance."""
        distance = self.classifier.label_space.distance(label_a, label_b)
        return distance <= self.tolerance

    def check_sufficiency(
        self,
        original_content: str,
        summary: str,
        rubric: str,
        node_id: Optional[str] = None,
    ) -> LawCheckResult:
        """
        Check C1 (Sufficiency): Does summary preserve oracle-relevant info?

        Compares oracle(original) with oracle(summary). If they differ beyond
        tolerance, the summary has lost information needed to compute the oracle.

        Args:
            original_content: The original text
            summary: The summary of the original
            rubric: Description of what to preserve
            node_id: Optional identifier for the node

        Returns:
            LawCheckResult with pass/fail and discrepancy
        """
        # Get predictions for both
        orig_pred = self.classifier(
            original_content=original_content,
            summary=original_content,  # Classify original directly
            rubric=rubric,
        )
        summ_pred = self.classifier(
            original_content=original_content,
            summary=summary,
            rubric=rubric,
        )

        # Compare predictions
        discrepancy = self.classifier.label_space.distance(orig_pred.label, summ_pred.label)
        passed = discrepancy <= self.tolerance

        reasoning = None
        if not passed:
            reasoning = (
                f"Sufficiency violation: Original predicted '{orig_pred.label}' "
                f"but summary predicted '{summ_pred.label}' (discrepancy={discrepancy:.2f}). "
                f"Original reasoning: {orig_pred.reasoning[:200]}... "
                f"Summary reasoning: {summ_pred.reasoning[:200]}..."
            )

        return LawCheckResult(
            law="sufficiency",
            passed=passed,
            discrepancy=discrepancy,
            original_prediction=orig_pred,
            summary_prediction=summ_pred,
            node_id=node_id,
            reasoning=reasoning,
        )

    def check_idempotence(
        self,
        summary: str,
        rubric: str,
        re_summary: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> LawCheckResult:
        """
        Check C2 (Idempotence): Does re-summarizing change the oracle?

        A stable summary should produce the same oracle prediction even after
        being summarized again. This checks that the summary doesn't contain
        extraneous detail that could be interpreted differently.

        Args:
            summary: The summary to check
            rubric: Description of what to preserve
            re_summary: Pre-computed re-summarization (if None, uses self.summarizer)
            node_id: Optional identifier for the node

        Returns:
            LawCheckResult with pass/fail and discrepancy
        """
        # Get re-summary if not provided
        if re_summary is None:
            if self.summarizer is None:
                # Can't check idempotence without a summarizer
                return LawCheckResult(
                    law="idempotence",
                    passed=True,
                    discrepancy=0.0,
                    node_id=node_id,
                    reasoning="Skipped: no summarizer provided for idempotence check",
                )
            re_summary = self.summarizer(summary)

        # Get predictions
        summ_pred = self.classifier(
            original_content=summary,
            summary=summary,
            rubric=rubric,
        )
        re_pred = self.classifier(
            original_content=summary,
            summary=re_summary,
            rubric=rubric,
        )

        # Compare predictions
        discrepancy = self.classifier.label_space.distance(summ_pred.label, re_pred.label)
        passed = discrepancy <= self.tolerance

        reasoning = None
        if not passed:
            reasoning = (
                f"Idempotence violation: Summary predicted '{summ_pred.label}' "
                f"but re-summary predicted '{re_pred.label}' (discrepancy={discrepancy:.2f}). "
                f"The summary is not stable under further summarization."
            )

        return LawCheckResult(
            law="idempotence",
            passed=passed,
            discrepancy=discrepancy,
            original_prediction=summ_pred,
            summary_prediction=re_pred,
            node_id=node_id,
            reasoning=reasoning,
        )

    def check_merge_consistency(
        self,
        merged_summary: str,
        child_summaries: List[str],
        rubric: str,
        child_weights: Optional[List[float]] = None,
        node_id: Optional[str] = None,
    ) -> LawCheckResult:
        """
        Check C3B (Merge Consistency): Is merged result consistent with children?

        When child summaries are merged, the resulting summary should produce
        an oracle prediction consistent with the aggregated predictions of
        the children.

        Args:
            merged_summary: The summary produced by merging children
            child_summaries: List of child summaries that were merged
            rubric: Description of what to preserve
            child_weights: Optional weights for children (e.g., by text length)
            node_id: Optional identifier for the node

        Returns:
            LawCheckResult with pass/fail and discrepancy
        """
        # Get prediction for merged summary
        merged_pred = self.classifier(
            original_content=merged_summary,
            summary=merged_summary,
            rubric=rubric,
        )

        # Get predictions for each child
        child_preds = []
        for child in child_summaries:
            pred = self.classifier(
                original_content=child,
                summary=child,
                rubric=rubric,
            )
            child_preds.append(pred)

        # Aggregate child predictions
        expected_label = self._aggregate_predictions(
            child_preds,
            weights=child_weights,
        )

        # Compare
        discrepancy = self.classifier.label_space.distance(merged_pred.label, expected_label)
        passed = discrepancy <= self.tolerance

        reasoning = None
        if not passed:
            child_labels = [p.label for p in child_preds]
            reasoning = (
                f"Merge consistency violation: Children predicted {child_labels} "
                f"(aggregated to '{expected_label}') but merged summary predicted "
                f"'{merged_pred.label}' (discrepancy={discrepancy:.2f})."
            )

        return LawCheckResult(
            law="merge_consistency",
            passed=passed,
            discrepancy=discrepancy,
            original_prediction=None,  # Not applicable for merge
            summary_prediction=merged_pred,
            expected_label=expected_label,
            node_id=node_id,
            reasoning=reasoning,
        )

    def _aggregate_predictions(
        self,
        child_preds: List[Prediction],
        weights: Optional[List[float]] = None,
    ) -> str:
        """
        Aggregate child predictions for merge consistency check.

        Strategy depends on label space type:
        - Ordinal: Weighted average of label values
        - Categorical: Most "severe" label (any non-none → that violation)
        """
        if not child_preds:
            return self.classifier.label_space.labels[0]

        if weights is None:
            weights = [p.confidence for p in child_preds]

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0] * len(child_preds)
            total_weight = len(child_preds)

        if self.classifier.label_space.is_ordinal:
            # Weighted average for ordinal
            try:
                values = [float(p.label) for p in child_preds]
                weighted_sum = sum(v * w for v, w in zip(values, weights))
                avg_value = weighted_sum / total_weight

                # Return nearest label
                from .core import OrdinalLabelSpace
                if isinstance(self.classifier.label_space, OrdinalLabelSpace):
                    return self.classifier.label_space.nearest_label(avg_value)
                else:
                    return str(int(round(avg_value)))
            except ValueError:
                return child_preds[0].label
        else:
            # Categorical: return most severe (non-none if any)
            non_none = [p for p in child_preds if p.label.lower() != "none"]
            if non_none:
                # Return highest confidence non-none
                return max(non_none, key=lambda p: p.confidence).label
            return "none"

    def check_substitution(
        self,
        summary_a: str,
        summary_b: str,
        rubric: str,
        node_id: Optional[str] = None,
    ) -> LawCheckResult:
        """
        Check C3A (Substitution): Do equivalent summaries produce same oracle?

        If two summaries are semantically equivalent, they should produce
        the same oracle prediction. This catches cases where the oracle
        is sensitive to surface-level differences.

        Args:
            summary_a: First summary
            summary_b: Second summary (should be equivalent to first)
            rubric: Description of what to preserve
            node_id: Optional identifier for the node

        Returns:
            LawCheckResult with pass/fail and discrepancy
        """
        # Get predictions for both
        pred_a = self.classifier(
            original_content=summary_a,
            summary=summary_a,
            rubric=rubric,
        )
        pred_b = self.classifier(
            original_content=summary_b,
            summary=summary_b,
            rubric=rubric,
        )

        # Compare predictions
        discrepancy = self.classifier.label_space.distance(pred_a.label, pred_b.label)
        passed = discrepancy <= self.tolerance

        reasoning = None
        if not passed:
            reasoning = (
                f"Substitution violation: Equivalent summaries produced different "
                f"predictions: '{pred_a.label}' vs '{pred_b.label}' "
                f"(discrepancy={discrepancy:.2f}). "
                f"The oracle is sensitive to surface-level differences."
            )

        return LawCheckResult(
            law="substitution",
            passed=passed,
            discrepancy=discrepancy,
            original_prediction=pred_a,
            summary_prediction=pred_b,
            node_id=node_id,
            reasoning=reasoning,
        )

    def verify_node(
        self,
        original_content: str,
        summary: str,
        rubric: str,
        child_summaries: Optional[List[str]] = None,
        re_summary: Optional[str] = None,
        node_id: Optional[str] = None,
        checks: Optional[List[str]] = None,
    ) -> NodeVerificationResult:
        """
        Run all applicable law checks for a node.

        Args:
            original_content: Original text (for leaf nodes)
            summary: The summary at this node
            rubric: Description of what to preserve
            child_summaries: Children summaries (for internal nodes)
            re_summary: Pre-computed re-summarization (for idempotence)
            node_id: Identifier for the node
            checks: Which checks to run (default: all applicable)

        Returns:
            NodeVerificationResult with all check results
        """
        if node_id is None:
            node_id = f"node_{uuid.uuid4().hex[:8]}"

        results = {}
        is_leaf = child_summaries is None or len(child_summaries) == 0

        # Determine which checks to run
        if checks is None:
            checks = ["sufficiency", "idempotence"]
            if not is_leaf:
                checks.append("merge_consistency")

        # Run checks
        if "sufficiency" in checks and is_leaf:
            results["sufficiency"] = self.check_sufficiency(
                original_content, summary, rubric, node_id
            )

        if "idempotence" in checks:
            results["idempotence"] = self.check_idempotence(
                summary, rubric, re_summary, node_id
            )

        if "merge_consistency" in checks and child_summaries:
            results["merge_consistency"] = self.check_merge_consistency(
                summary, child_summaries, rubric, node_id=node_id
            )

        return NodeVerificationResult(node_id=node_id, law_results=results)


class TreeVerifier:
    """
    Verifies OPS laws across an entire summarization tree.

    Walks the tree and runs appropriate checks at each node,
    collecting training data from violations.
    """

    def __init__(
        self,
        classifier: OracleClassifier,
        tolerance: float = 0.0,
        summarizer: Optional[Callable[[str], str]] = None,
    ):
        self.node_verifier = OracleNodeVerifier(classifier, tolerance, summarizer)
        self.results: List[NodeVerificationResult] = []

    def verify_tree(
        self,
        tree_data: Dict,
        rubric: str,
    ) -> Dict[str, NodeVerificationResult]:
        """
        Verify all nodes in a tree.

        Args:
            tree_data: Tree structure with nodes containing:
                - 'id': Node identifier
                - 'original': Original content (for leaves)
                - 'summary': Summary at this node
                - 'children': List of child node dicts (for internal nodes)
            rubric: Description of what to preserve

        Returns:
            Dict mapping node_id to verification result
        """
        results = {}
        self._verify_node_recursive(tree_data, rubric, results)
        return results

    def _verify_node_recursive(
        self,
        node: Dict,
        rubric: str,
        results: Dict[str, NodeVerificationResult],
    ):
        """Recursively verify nodes."""
        node_id = node.get('id', f"node_{uuid.uuid4().hex[:8]}")
        summary = node.get('summary', '')
        children = node.get('children', [])

        # Recursively verify children first
        child_summaries = []
        for child in children:
            self._verify_node_recursive(child, rubric, results)
            child_summaries.append(child.get('summary', ''))

        # Verify this node
        original = node.get('original', summary)  # Use summary if no original
        result = self.node_verifier.verify_node(
            original_content=original,
            summary=summary,
            rubric=rubric,
            child_summaries=child_summaries if children else None,
            node_id=node_id,
        )

        results[node_id] = result
        self.results.append(result)

    def get_training_data(
        self,
        tree_data: Dict,
        rubric: str,
    ) -> List[UnifiedTrainingExample]:
        """
        Extract training examples from tree verification.

        Returns both positive (violations) and negative (passes) examples.
        """
        # Verify tree
        results = self.verify_tree(tree_data, rubric)

        # Collect training examples
        examples = []
        for node_id, result in results.items():
            # Get node data
            node = self._find_node(tree_data, node_id)
            if node:
                original = node.get('original', node.get('summary', ''))
                summary = node.get('summary', '')
                examples.extend(result.to_training_examples(original, summary, rubric))

        return examples

    def _find_node(self, tree: Dict, node_id: str) -> Optional[Dict]:
        """Find a node by ID in the tree."""
        if tree.get('id') == node_id:
            return tree
        for child in tree.get('children', []):
            found = self._find_node(child, node_id)
            if found:
                return found
        return None

    def get_statistics(self) -> Dict:
        """Get verification statistics."""
        total_checks = 0
        passed_checks = 0
        by_law = {}

        for result in self.results:
            for law, check in result.law_results.items():
                total_checks += 1
                if check.passed:
                    passed_checks += 1

                if law not in by_law:
                    by_law[law] = {'total': 0, 'passed': 0}
                by_law[law]['total'] += 1
                if check.passed:
                    by_law[law]['passed'] += 1

        return {
            'total_nodes': len(self.results),
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'pass_rate': passed_checks / total_checks if total_checks > 0 else 1.0,
            'by_law': {
                law: {
                    'total': stats['total'],
                    'passed': stats['passed'],
                    'pass_rate': stats['passed'] / stats['total'] if stats['total'] > 0 else 1.0,
                }
                for law, stats in by_law.items()
            },
        }
