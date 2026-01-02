"""
OPS Bootstrap Training Loop (Paper Section 3.11).

This module implements the full audit → collect failures → retrain cycle
that optimizes summarizers to satisfy OPS laws:

    1. Process documents with current summarizer
    2. Run probabilistic audit to estimate violation rates (p_suff, p_merge, p_idem)
    3. Collect training examples from audit failures
    4. Retrain summarizer using DSPy optimization
    5. Repeat until violation rates drop below threshold OR convergence

The key insight from the paper is that we can train on summaries (not full docs)
when OPS conditions hold, which is Theorem 3.1's DPO equivalence.

Usage:
    from src.ops_engine.bootstrap_loop import BootstrapTrainer, BootstrapConfig

    trainer = BootstrapTrainer(
        oracle=my_oracle,
        summarizer=my_summarizer,
        config=BootstrapConfig(target_p_suff=0.05),
    )

    result = trainer.train(documents, rubric)
    print(f"Final p_suff: {result.final_violation_rates['p_suff']}")
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

if TYPE_CHECKING:
    from src.core.data_models import Node

logger = logging.getLogger(__name__)


# =============================================================================
# Tree Normalization Helpers
# =============================================================================

def _normalize_to_node(obj: Any) -> 'Node':
    """
    Normalize any tree representation to a Node object.

    Handles:
    - Node objects (passthrough)
    - Tree objects (extracts root)
    - BuildResult objects (extracts tree.root)
    - Dict trees (converts to Node)

    Args:
        obj: Any tree-like object

    Returns:
        Node object
    """
    from src.core.data_models import Node, leaf, node as make_node

    # Already a Node - passthrough
    if isinstance(obj, Node):
        return obj

    # Handle BuildResult from builder.py
    if hasattr(obj, 'tree'):
        return _normalize_to_node(obj.tree)

    # Handle Tree (has .root attribute)
    if hasattr(obj, 'root'):
        return _normalize_to_node(obj.root)

    # Handle dict representation - convert to Node
    if isinstance(obj, dict):
        node_id = obj.get('id', 'unknown')
        text = obj.get('text', obj.get('raw_text_span', ''))
        summary = obj.get('summary', '')
        is_leaf = obj.get('is_leaf', True)
        children = obj.get('children', [])

        if is_leaf or not children:
            # Create leaf node
            result = leaf(text, node_id=node_id)
            result.summary = summary
            return result
        else:
            # Create internal node with children
            child_nodes = [_normalize_to_node(c) for c in children]
            # For binary trees, assign left/right
            left = child_nodes[0] if len(child_nodes) > 0 else None
            right = child_nodes[1] if len(child_nodes) > 1 else None
            result = make_node(left, right, summary=summary, node_id=node_id)
            return result

    # Fallback - create a leaf node from object attributes
    text = getattr(obj, 'text', getattr(obj, 'raw_text_span', ''))
    node_id = getattr(obj, 'id', 'unknown')
    summary = getattr(obj, 'summary', '')
    result = leaf(text, node_id=node_id)
    result.summary = summary
    return result


@dataclass
class BootstrapConfig:
    """Configuration for the OPS bootstrap training loop."""

    # Convergence thresholds (target violation rates from paper)
    target_p_suff: float = 0.05      # Target sufficiency violation rate
    target_p_merge: float = 0.05     # Target merge consistency violation rate
    target_p_idem: float = 0.10      # Target idempotence violation rate (can be higher)

    # Convergence criteria
    max_iterations: int = 10         # Maximum bootstrap iterations
    convergence_threshold: float = 0.01  # Stop if improvement < this
    convergence_patience: int = 2    # Stop after N iters without improvement

    # Sampling parameters (probabilistic audit)
    sample_rate: float = 0.10        # Sample 10% of nodes for audit
    min_samples: int = 10            # Minimum samples per check type

    # Training data collection
    max_training_examples: int = 100  # Max examples to collect per iteration
    balance_positive_negative: bool = True  # Balance violations vs. good examples
    include_near_misses: bool = True  # Include examples close to threshold

    # DSPy optimization settings
    optimizer_type: str = "bootstrap_random_search"
    optimizer_budget: str = "medium"
    num_threads: int = 64

    # Checkpointing
    checkpoint_dir: Optional[Path] = None
    save_intermediate: bool = True


@dataclass
class BootstrapIteration:
    """Results from a single bootstrap iteration."""
    iteration: int
    violation_rates: Dict[str, float]  # p_suff, p_merge, p_idem, union_bound
    training_examples_collected: int
    optimization_score_before: float
    optimization_score_after: float
    duration_seconds: float

    @property
    def improvement(self) -> float:
        """Score improvement from optimization."""
        return self.optimization_score_after - self.optimization_score_before


@dataclass
class BootstrapResult:
    """Final result of the bootstrap training loop."""
    converged: bool
    iterations_run: int
    final_violation_rates: Dict[str, float]
    iteration_history: List[BootstrapIteration]
    total_training_examples: int
    total_duration_seconds: float

    # Target thresholds (defaults match BootstrapConfig defaults)
    target_p_suff: float = 0.05
    target_p_merge: float = 0.05
    target_p_idem: float = 0.10

    @property
    def meets_targets(self) -> bool:
        """Whether all violation rates meet configured targets."""
        return (
            self.final_violation_rates.get("p_suff", 1.0) <= self.target_p_suff and
            self.final_violation_rates.get("p_merge", 1.0) <= self.target_p_merge and
            self.final_violation_rates.get("p_idem", 1.0) <= self.target_p_idem
        )


# Import UnifiedTrainingExample for audit examples (replaces AuditTrainingExample)
from src.ops_engine.training_framework.core import UnifiedTrainingExample


class BootstrapTrainer:
    """
    Implements the full OPS bootstrap training loop from paper Section 3.11.

    The loop alternates between:
    1. Auditing: Estimate violation rates via probabilistic sampling
    2. Training: Optimize summarizer on collected failure examples

    This continues until violation rates meet targets or convergence.
    """

    def __init__(
        self,
        oracle_fn: Union[Callable[[str, str, str], Tuple[bool, float, str]], Any],
        summarizer_fn: Callable[[str, str], str],
        config: Optional[BootstrapConfig] = None,
        tree_builder_fn: Optional[Callable] = None,
    ):
        """
        Initialize the bootstrap trainer.

        Args:
            oracle_fn: Oracle function - either ScoringOracle (preferred) or legacy
                      (input_a, input_b, rubric) -> (congruent, discrepancy, reasoning)
            summarizer_fn: Summarizer function (content, rubric) -> summary
            config: Bootstrap configuration
            tree_builder_fn: Optional function to build trees from documents
        """
        self._oracle = oracle_fn
        self.summarizer = summarizer_fn
        self.config = config or BootstrapConfig()
        self.tree_builder = tree_builder_fn

        # Track training data across iterations
        self._all_training_examples: List[UnifiedTrainingExample] = []
        self._iteration_history: List[BootstrapIteration] = []

    @property
    def oracle(self):
        """Access the oracle function."""
        return self._oracle

    def _call_oracle(self, input_a: str, input_b: str, rubric: str) -> Tuple[bool, float, str]:
        """
        Call the oracle and return (is_congruent, discrepancy, reasoning).

        Requires ScoringOracle interface (has .score() method returning OracleScore).
        """
        result = self._oracle.score(input_a, input_b, rubric)
        # Score is 0.0-1.0 where 1.0 = good; discrepancy is inverse
        discrepancy = 1.0 - result.score
        is_congruent = discrepancy <= self.config.target_p_suff
        return is_congruent, discrepancy, result.reasoning

    def train(
        self,
        documents: List[Dict[str, Any]],
        rubric: str,
        dspy_module: Optional[Any] = None,
    ) -> BootstrapResult:
        """
        Run the full bootstrap training loop.

        Args:
            documents: List of documents to process (each has 'text' and optional 'id')
            rubric: The information preservation rubric
            dspy_module: Optional DSPy module to optimize (e.g., LeafSummarizer)

        Returns:
            BootstrapResult with final violation rates and training history
        """
        start_time = time.time()
        logger.info(f"Starting OPS bootstrap loop (max {self.config.max_iterations} iterations)")
        logger.info(f"Targets: p_suff≤{self.config.target_p_suff}, "
                   f"p_merge≤{self.config.target_p_merge}, p_idem≤{self.config.target_p_idem}")

        prev_union_bound = 1.0
        no_improvement_count = 0
        converged = False

        for iteration in range(1, self.config.max_iterations + 1):
            iter_start = time.time()
            logger.info(f"\n{'='*60}")
            logger.info(f"Bootstrap Iteration {iteration}/{self.config.max_iterations}")
            logger.info(f"{'='*60}")

            # Step 1: Process documents and build trees
            logger.info("Step 1: Processing documents...")
            trees = self._build_trees(documents, rubric)
            logger.info(f"  Built {len(trees)} trees")

            # Step 2: Run probabilistic audit
            logger.info("Step 2: Running probabilistic audit...")
            violation_rates, examples = self._run_audit(trees, rubric)

            logger.info(f"  Violation rates:")
            logger.info(f"    p_suff:  {violation_rates.get('p_suff', 0):.4f}")
            logger.info(f"    p_merge: {violation_rates.get('p_merge', 0):.4f}")
            logger.info(f"    p_idem:  {violation_rates.get('p_idem', 0):.4f}")
            logger.info(f"    Union bound: {violation_rates.get('union_bound', 0):.4f}")

            # Step 3: Check if we've met targets
            if self._meets_targets(violation_rates):
                logger.info("✓ All violation rates meet targets! Stopping.")
                converged = True
                iter_result = BootstrapIteration(
                    iteration=iteration,
                    violation_rates=violation_rates,
                    training_examples_collected=len(examples),
                    optimization_score_before=1.0 - violation_rates.get('union_bound', 0),
                    optimization_score_after=1.0 - violation_rates.get('union_bound', 0),
                    duration_seconds=time.time() - iter_start,
                )
                self._iteration_history.append(iter_result)
                break

            # Step 4: Collect training examples from failures
            logger.info(f"Step 3: Collecting training examples...")
            training_examples = self._collect_training_examples(examples)
            self._all_training_examples.extend(training_examples)
            logger.info(f"  Collected {len(training_examples)} new examples "
                       f"({len(self._all_training_examples)} total)")

            # Step 5: Optimize summarizer
            score_before = 1.0 - violation_rates.get('union_bound', 1.0)
            if dspy_module is not None and training_examples:
                logger.info("Step 4: Optimizing summarizer with DSPy...")
                score_after = self._optimize_summarizer(
                    dspy_module,
                    training_examples,
                    rubric,
                )
                if score_after < 0:
                    logger.warning(f"  Optimization failed (score={score_after}), using previous model")
                    score_after = score_before  # Keep previous score on failure
                else:
                    logger.info(f"  Optimization score: {score_before:.3f} → {score_after:.3f}")
            else:
                score_after = score_before
                if not training_examples:
                    logger.info("Step 4: Skipping optimization (no training examples)")
                else:
                    logger.info("Step 4: Skipping optimization (no DSPy module provided)")

            # Step 6: Check convergence
            current_bound = violation_rates.get('union_bound', 1.0)
            improvement = prev_union_bound - current_bound

            if improvement < self.config.convergence_threshold:
                no_improvement_count += 1
                logger.info(f"  No significant improvement ({no_improvement_count}/{self.config.convergence_patience})")
            else:
                no_improvement_count = 0
                logger.info(f"  Improvement: {improvement:.4f}")

            prev_union_bound = current_bound

            # Record iteration
            iter_result = BootstrapIteration(
                iteration=iteration,
                violation_rates=violation_rates,
                training_examples_collected=len(training_examples),
                optimization_score_before=score_before,
                optimization_score_after=score_after,
                duration_seconds=time.time() - iter_start,
            )
            self._iteration_history.append(iter_result)

            # Save checkpoint if configured
            if self.config.save_intermediate and self.config.checkpoint_dir:
                self._save_checkpoint(iteration, violation_rates)

            # Check for early stopping
            if no_improvement_count >= self.config.convergence_patience:
                logger.info(f"Converged (no improvement for {self.config.convergence_patience} iterations)")
                converged = True
                break

        # Final result
        final_rates = self._iteration_history[-1].violation_rates if self._iteration_history else {}

        result = BootstrapResult(
            converged=converged,
            iterations_run=len(self._iteration_history),
            final_violation_rates=final_rates,
            iteration_history=self._iteration_history,
            total_training_examples=len(self._all_training_examples),
            total_duration_seconds=time.time() - start_time,
            # Pass targets from config so meets_targets() uses correct values
            target_p_suff=self.config.target_p_suff,
            target_p_merge=self.config.target_p_merge,
            target_p_idem=self.config.target_p_idem,
        )

        logger.info(f"\n{'='*60}")
        logger.info("Bootstrap Training Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Iterations: {result.iterations_run}")
        logger.info(f"Converged: {result.converged}")
        logger.info(f"Meets targets: {result.meets_targets}")
        logger.info(f"Final union bound: {result.final_violation_rates.get('union_bound', 'N/A')}")
        logger.info(f"Total time: {result.total_duration_seconds:.1f}s")

        return result

    def _build_trees(
        self,
        documents: List[Dict[str, Any]],
        rubric: str,
    ) -> List['Node']:
        """
        Build summarization trees for all documents.

        Returns list of Node objects (root nodes of each tree).
        Works with any tree_builder that returns Node, Tree, BuildResult, or dict.
        """
        from src.core.data_models import leaf

        if self.tree_builder is not None:
            # Build trees using provided builder and normalize to Node
            raw_trees = [self.tree_builder(doc, rubric) for doc in documents]
            return [_normalize_to_node(tree) for tree in raw_trees]

        # Simple fallback: create single-node trees as Node objects
        trees = []
        for i, doc in enumerate(documents):
            text = doc.get('text', str(doc))
            doc_id = doc.get('id', f'doc_{i}')
            summary = self.summarizer(text, rubric)
            node = leaf(text, node_id=doc_id)
            node.summary = summary
            trees.append(node)
        return trees

    def _run_audit(
        self,
        trees: List['Node'],
        rubric: str,
    ) -> Tuple[Dict[str, float], List[UnifiedTrainingExample]]:
        """
        Run probabilistic audit and collect violation examples.

        Args:
            trees: List of Node objects (root nodes of each tree)
            rubric: The information preservation rubric

        Returns:
            Tuple of (violation_rates dict, list of examples)
        """
        from src.ops_engine.checks import CheckRunner, CheckConfig, CheckType

        check_config = CheckConfig(discrepancy_threshold=0.1)
        runner = CheckRunner(
            oracle_fn=self.oracle,
            config=check_config,
            summarizer_fn=self.summarizer,
        )

        examples = []
        counts = {
            'sufficiency_total': 0, 'sufficiency_violations': 0,
            'merge_total': 0, 'merge_violations': 0,
            'idempotence_total': 0, 'idempotence_violations': 0,
        }
        total_leaves = 0
        total_merges = 0

        for tree in trees:
            tree_id = tree.id

            # Count structure
            if tree.is_leaf:
                total_leaves += 1
            else:
                total_merges += 1
                total_leaves += len(tree.children)

            # Sample for audit based on sample_rate
            import random
            if random.random() > self.config.sample_rate:
                continue

            # Check sufficiency for leaves
            if tree.is_leaf and tree.raw_text_span:
                result = runner.check_sufficiency(
                    original_content=tree.raw_text_span,
                    summary=tree.summary or '',
                    rubric=rubric,
                    node_id=tree_id,
                )
                counts['sufficiency_total'] += 1
                if not result.passed:
                    counts['sufficiency_violations'] += 1
                    examples.append(UnifiedTrainingExample.create_audit_example(
                        example_id=f"{tree_id}_suff",
                        check_type="sufficiency",
                        is_violation=True,
                        original_content=tree.raw_text_span,
                        current_summary=tree.summary or '',
                        rubric=rubric,
                        discrepancy=result.discrepancy,
                        node_id=tree_id,
                        document_id=tree_id,
                    ))

            # Check idempotence
            if tree.summary:
                result = runner.check_idempotence(
                    summary=tree.summary,
                    rubric=rubric,
                    node_id=tree_id,
                )
                counts['idempotence_total'] += 1
                if not result.passed:
                    counts['idempotence_violations'] += 1

            # Check merge consistency for internal nodes
            children = tree.children
            if children and len(children) >= 2:
                child_summaries = [c.summary for c in children if c.summary]
                if child_summaries:
                    result = runner.check_merge_consistency(
                        child_summaries=child_summaries,
                        parent_summary=tree.summary or '',
                        rubric=rubric,
                        node_id=tree_id,
                    )
                    counts['merge_total'] += 1
                    if not result.passed:
                        counts['merge_violations'] += 1

        # Compute violation rates
        p_suff = (counts['sufficiency_violations'] / counts['sufficiency_total']
                  if counts['sufficiency_total'] > 0 else 0.0)
        p_merge = (counts['merge_violations'] / counts['merge_total']
                   if counts['merge_total'] > 0 else 0.0)
        p_idem = (counts['idempotence_violations'] / counts['idempotence_total']
                  if counts['idempotence_total'] > 0 else 0.0)

        # Compute union bound (Equation 1 from paper)
        # Pr[root violation] ≤ N*p_suff + M*p_merge + (R-1)*p_idem
        # Simplified: assume R=1 (single round)
        union_bound = min(1.0, total_leaves * p_suff + total_merges * p_merge)

        violation_rates = {
            'p_suff': p_suff,
            'p_merge': p_merge,
            'p_idem': p_idem,
            'union_bound': union_bound,
            'total_leaves': total_leaves,
            'total_merges': total_merges,
            'samples': counts,
        }

        return violation_rates, examples

    def _meets_targets(self, violation_rates: Dict[str, float]) -> bool:
        """Check if all violation rates meet targets."""
        return (
            violation_rates.get('p_suff', 1.0) <= self.config.target_p_suff and
            violation_rates.get('p_merge', 1.0) <= self.config.target_p_merge and
            violation_rates.get('p_idem', 1.0) <= self.config.target_p_idem
        )

    def _collect_training_examples(
        self,
        audit_examples: List[UnifiedTrainingExample],
    ) -> List[UnifiedTrainingExample]:
        """
        Filter and balance training examples from audit.

        Applies:
        - Max examples limit
        - Positive/negative balancing if configured
        """
        from src.ops_engine.training_framework.core import TrainingExampleLabel

        if not audit_examples:
            return []

        # Separate violations and good examples
        violations = [ex for ex in audit_examples if ex.label == TrainingExampleLabel.POSITIVE]
        good = [ex for ex in audit_examples if ex.label == TrainingExampleLabel.NEGATIVE]

        if self.config.balance_positive_negative and violations and good:
            # Balance to smaller class
            min_count = min(len(violations), len(good))
            import random
            violations = random.sample(violations, min(min_count, len(violations)))
            good = random.sample(good, min(min_count, len(good)))

        combined = violations + good

        # Apply max limit
        if len(combined) > self.config.max_training_examples:
            import random
            combined = random.sample(combined, self.config.max_training_examples)

        return combined

    def _optimize_summarizer(
        self,
        dspy_module: Any,
        training_examples: List[UnifiedTrainingExample],
        rubric: str,
    ) -> float:
        """
        Optimize the DSPy summarizer module using collected examples.

        Returns:
            Final optimization score
        """
        try:
            import dspy

            # Convert to DSPy examples
            dspy_examples = []
            for ex in training_examples:
                dspy_ex = dspy.Example(
                    content=ex.original_content,
                    rubric=rubric,
                    summary=ex.current_summary,
                    is_violation=ex.is_violation,
                    discrepancy=ex.discrepancy,
                ).with_inputs('content', 'rubric')
                dspy_examples.append(dspy_ex)

            if not dspy_examples:
                return 0.0

            # Create metric that rewards low discrepancy
            def summarization_metric(gold, pred, trace=None):
                """Metric: 1.0 for good summaries, lower for violations."""
                pred_summary = getattr(pred, 'summary', str(pred))
                orig_content = getattr(gold, 'content', '')

                # Use oracle to score
                is_congruent, discrepancy, _ = self._call_oracle(
                    orig_content, pred_summary, rubric
                )

                # Score: 1.0 - discrepancy (clamped to 0-1)
                score = max(0.0, 1.0 - discrepancy)
                return score

            # Create optimizer
            if self.config.optimizer_type == 'grpo':
                try:
                    optimizer = dspy.GRPO(
                        metric=summarization_metric,
                        num_threads=self.config.num_threads,
                    )
                except AttributeError:
                    optimizer = dspy.BootstrapFewShotWithRandomSearch(
                        metric=summarization_metric,
                        max_bootstrapped_demos=4,
                        num_candidate_programs=8,
                        num_threads=self.config.num_threads,
                    )
            else:
                optimizer = dspy.BootstrapFewShotWithRandomSearch(
                    metric=summarization_metric,
                    max_bootstrapped_demos=4,
                    num_candidate_programs=8,
                    num_threads=self.config.num_threads,
                )

            # Run optimization
            compiled = optimizer.compile(
                student=dspy_module,
                trainset=dspy_examples,
            )

            # DSPy stores scores on the compiled program, not the optimizer
            if hasattr(compiled, 'candidate_programs') and compiled.candidate_programs:
                score = compiled.candidate_programs[0].get('score', -1.0)
            else:
                logger.warning("Could not retrieve score - candidate_programs not found")
                score = -1.0
            return score

        except ImportError:
            logger.warning("DSPy not available, skipping optimization")
            return -1.0  # Negative score indicates failure
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return -1.0  # Negative score indicates failure

    def _save_checkpoint(self, iteration: int, violation_rates: Dict[str, float]):
        """Save iteration checkpoint."""
        if not self.config.checkpoint_dir:
            return

        import json
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = checkpoint_dir / f"bootstrap_iter_{iteration}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'iteration': iteration,
                'violation_rates': violation_rates,
                'training_examples_count': len(self._all_training_examples),
            }, f, indent=2)


# =============================================================================
# Convenience Functions
# =============================================================================

def run_bootstrap_training(
    documents: List[Dict[str, Any]],
    rubric: str,
    oracle_fn: Callable,
    summarizer_fn: Callable,
    dspy_module: Any = None,
    max_iterations: int = 5,
    target_p_suff: float = 0.05,
    checkpoint_dir: Optional[Path] = None,
) -> BootstrapResult:
    """
    Convenience function to run bootstrap training.

    Args:
        documents: List of documents (each with 'text' key)
        rubric: Information preservation rubric
        oracle_fn: Oracle function
        summarizer_fn: Summarizer function
        dspy_module: Optional DSPy module to optimize
        max_iterations: Maximum bootstrap iterations
        target_p_suff: Target sufficiency violation rate
        checkpoint_dir: Optional directory for checkpoints

    Returns:
        BootstrapResult with training outcomes
    """
    config = BootstrapConfig(
        max_iterations=max_iterations,
        target_p_suff=target_p_suff,
        checkpoint_dir=checkpoint_dir,
    )

    trainer = BootstrapTrainer(
        oracle_fn=oracle_fn,
        summarizer_fn=summarizer_fn,
        config=config,
    )

    return trainer.train(documents, rubric, dspy_module)
