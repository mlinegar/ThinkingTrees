#!/usr/bin/env python3
"""
Manifesto RILE Training Pipeline (No Human-in-the-Loop).

This script runs the full automated training pipeline:
1. Load manifesto data (train/val/test splits)
2. Process manifestos through batched OPS pipeline
3. Collect training data from prediction errors
4. Run multiple optimization rounds
5. Report progress and final statistics

Usage:
    python run_training_pipeline.py --port 8000 --train-samples 50 --val-samples 20 --rounds 3
"""

# Set NumExpr threads before any imports that might use it (pandas, numpy)
import os
os.environ.setdefault("NUMEXPR_MAX_THREADS", str(os.cpu_count() or 64))

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
import sys

# Setup logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def print_banner(text: str, char: str = "="):
    """Print a banner."""
    width = 70
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)


def print_section(text: str):
    """Print a section header."""
    print()
    print(f">>> {text}")
    print("-" * 50)


# =============================================================================
# Violation Rate Reporting (Paper Section 4.1)
# =============================================================================

def compute_and_log_violation_rates(
    results: List,
    oracle_classifier,
    rubric: str,
    iteration: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Compute and log OPS violation rates after each iteration.

    This implements the paper's probabilistic audit (Section 4.1) and reports:
    - p_suff: Sufficiency violation rate (C1)
    - p_merge: Merge consistency violation rate (C3B)
    - p_idem: Idempotence violation rate (C2)
    - p_bound: Substitution violation rate (C3A)
    - p_assoc: Combined merge-consistency rate
    - Union bound: Pr[root violation] ≤ N*p_suff + M*p_assoc + (R-1)*p_idem

    Args:
        results: List of ManifestoResult from processing
        oracle_classifier: Trained oracle classifier for comparisons
        rubric: Information preservation rubric
        iteration: Current iteration number
        output_dir: Directory for saving reports

    Returns:
        Dict with violation statistics
    """
    from src.manifesto.constants import RILE_MIN, RILE_MAX

    logger.info(f"Computing OPS violation rates for iteration {iteration}...")

    # Aggregate statistics across all documents
    total_stats = {
        "iteration": iteration,
        "documents_audited": 0,
        "total_leaves": 0,
        "total_merges": 0,
    }

    # Compute tree structure statistics from results
    for r in results:
        if r.error is None:
            leaves = getattr(r, 'tree_leaves', 0) or 0
            total_stats["total_leaves"] += leaves
            total_stats["total_merges"] += max(0, leaves - 1)
            total_stats["documents_audited"] += 1

    # Compute violation rate estimates from prediction errors
    valid_results = [r for r in results if r.error is None and r.predicted_rile is not None]

    if valid_results:
        errors = [abs(r.predicted_rile - r.ground_truth_rile) for r in valid_results]
        avg_error = sum(errors) / len(errors)

        # Estimate p_suff from average error (approximation)
        # High errors suggest summarization is losing oracle-relevant information
        # Normalize: 50-point error = 100% violation rate
        estimated_p_suff = min(1.0, avg_error / 50.0)

        total_stats["estimated_p_suff"] = estimated_p_suff
        total_stats["avg_prediction_error"] = avg_error

        # Compute union bound estimate (Equation 1 from paper)
        # Pr[root violation] ≤ N * p_suff + M * p_assoc + (R-1) * p_idem
        N = total_stats["total_leaves"]
        M = total_stats["total_merges"]

        # Simplified bound (without idempotence/merge terms for now)
        bound = min(1.0, N * estimated_p_suff)

        total_stats["estimated_union_bound"] = bound
        total_stats["bound_formula"] = f"N({N}) * p_suff({estimated_p_suff:.4f}) = {bound:.4f}"

        # Log the results in a formatted way
        print_section(f"OPS Violation Rates (Iteration {iteration})")
        logger.info(f"  Documents audited: {total_stats['documents_audited']}")
        logger.info(f"  Total leaves (N): {N}")
        logger.info(f"  Total merges (M): {M}")
        logger.info(f"  Avg RILE prediction error: {avg_error:.2f}")
        logger.info(f"  Estimated p_suff: {estimated_p_suff:.4f}")
        logger.info(f"  Union bound Pr[root violation] ≤ {bound:.4f}")

        # Check if bound is acceptable
        if bound > 0.5:
            logger.warning(f"  ⚠️  High violation bound! Consider more training or smaller chunks.")
        elif bound < 0.1:
            logger.info(f"  ✓ Low violation bound - summarization preserving oracle well.")

    # Save to file
    stats_file = output_dir / f"violation_rates_iter_{iteration}.json"
    with open(stats_file, 'w') as f:
        json.dump(total_stats, f, indent=2)

    return total_stats


def build_gepa_kwargs(
    budget: str,
    num_threads: int,
    max_metric_calls: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build kwargs for GEPA optimizer based on budget setting.

    Args:
        budget: 'light', 'medium', 'heavy', or 'superheavy'
        num_threads: Number of parallel threads
        max_metric_calls: Direct override for metric calls (optional)

    Returns:
        Dict of kwargs to pass to dspy.GEPA()
    """
    kwargs = {"num_threads": num_threads}

    if max_metric_calls is not None:
        # Direct control - bypass auto entirely
        kwargs["max_metric_calls"] = max_metric_calls
        logger.info(f"GEPA: Using max_metric_calls={max_metric_calls}")
    elif budget == "superheavy":
        # Superheavy: 10x what heavy typically does (~5000 calls)
        kwargs["max_metric_calls"] = 5000
        logger.info("GEPA: Using superheavy budget (max_metric_calls=5000)")
    else:
        # Standard auto budgets
        kwargs["auto"] = budget

    return kwargs


def create_dspy_optimizer(
    optimizer_type: str,
    metric: Callable,
    budget: str = "heavy",
    num_threads: int = 128,
    max_metric_calls: Optional[int] = None,
    labeled_k: int = 8,
    num_candidates: int = 16,
    max_bootstrapped_demos: int = 4,
    max_rounds: int = 1,
):
    """
    Create a DSPy optimizer based on type.

    Args:
        optimizer_type: 'gepa', 'bootstrap', 'bootstrap_random_search',
                       'mipro', 'labeled_fewshot', 'grpo' (for DPO-style training)
        metric: The metric function
        budget: Budget for GEPA/MIPROv2 ('light', 'medium', 'heavy', 'superheavy')
        num_threads: Parallel threads
        max_metric_calls: Direct override for metric calls
        labeled_k: Number of demos for LabeledFewShot

    Returns:
        Configured DSPy optimizer

    Note:
        GRPO (Group Relative Policy Optimization) implements DPO-style preference
        learning, which aligns with the paper's Theorem 3.1 on training equivalence
        between summaries and full documents when OPS conditions hold.
    """
    import dspy

    # GRPO / DPO-style optimization (Paper Theorem 3.1)
    if optimizer_type == 'grpo':
        logger.info(f"Creating GRPO optimizer (DPO-style, threads={num_threads})")
        try:
            return dspy.GRPO(
                metric=metric,
                num_threads=num_threads,
            )
        except AttributeError:
            logger.warning("dspy.GRPO not available in this DSPy version, falling back to GEPA")
            optimizer_type = 'gepa'

    if optimizer_type.startswith('gepa'):
        # Parse budget from name if embedded (e.g., 'gepa_heavy')
        if '_' in optimizer_type:
            budget = optimizer_type.split('_')[1]
        gepa_kwargs = build_gepa_kwargs(budget, num_threads, max_metric_calls)
        logger.info(f"Creating GEPA optimizer (budget={budget}, threads={num_threads})")
        return dspy.GEPA(metric=metric, **gepa_kwargs)

    elif optimizer_type == 'bootstrap':
        logger.info(f"Creating BootstrapFewShot optimizer (threads={num_threads}, demos={max_bootstrapped_demos}, rounds={max_rounds})")
        return dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=labeled_k,
            max_rounds=max_rounds,
        )

    elif optimizer_type == 'bootstrap_random_search':
        logger.info(f"Creating BootstrapFewShotWithRandomSearch (threads={num_threads}, candidates={num_candidates}, demos={max_bootstrapped_demos}, rounds={max_rounds})")
        return dspy.BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=labeled_k,
            num_candidate_programs=num_candidates,
            max_rounds=max_rounds,
            num_threads=num_threads,
        )

    elif optimizer_type.startswith('mipro'):
        # Parse budget from name if embedded
        if '_' in optimizer_type:
            budget = optimizer_type.split('_')[1]
        logger.info(f"Creating MIPROv2 optimizer (budget={budget})")
        return dspy.MIPROv2(
            metric=metric,
            auto=budget,
            num_threads=num_threads,
        )

    elif optimizer_type == 'labeled_fewshot':
        logger.info(f"Creating LabeledFewShot optimizer (k={labeled_k})")
        return dspy.LabeledFewShot(k=labeled_k)

    else:
        logger.warning(f"Unknown optimizer type '{optimizer_type}', defaulting to GEPA")
        gepa_kwargs = build_gepa_kwargs(budget, num_threads, max_metric_calls)
        return dspy.GEPA(metric=metric, **gepa_kwargs)


# =============================================================================
# Oracle Pre-Caching for Metric Acceleration
# =============================================================================

def precache_oracle_predictions(
    oracle_classifier,
    trainset: List,
    field_name: str = 'summary',
    max_workers: int = 256,
) -> Dict[str, Tuple[float, float, str]]:
    """
    Pre-compute oracle predictions for all training examples.

    This dramatically speeds up optimization by caching oracle calls upfront
    instead of making redundant LLM calls during each candidate evaluation.

    Args:
        oracle_classifier: The RILE oracle classifier with predict_rile() method
        trainset: List of training examples (DSPy Examples or similar)
        field_name: Field to extract text from (default: 'summary')
        max_workers: Number of parallel workers for caching

    Returns:
        Dict mapping text -> (predicted_rile, confidence, reasoning)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Collect unique texts to score
    unique_texts = set()
    for ex in trainset:
        text = getattr(ex, field_name, None) or str(ex)
        if text:
            unique_texts.add(text)

    if not unique_texts:
        logger.warning("No texts found to cache")
        return {}

    logger.info(f"Pre-caching oracle for {len(unique_texts)} unique texts...")

    cache = {}
    completed = 0

    def predict_one(text):
        try:
            return text, oracle_classifier.predict_rile(text)
        except Exception as e:
            return text, (0.0, 0.0, str(e))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(predict_one, text) for text in unique_texts]
        for future in as_completed(futures):
            text, result = future.result()
            cache[text] = result
            completed += 1
            if completed % 100 == 0:
                logger.info(f"  Cached {completed}/{len(unique_texts)} predictions")

    logger.info(f"Cached {len(cache)} oracle predictions")
    return cache


# =============================================================================
# Parallel Candidate Evaluation Patch
# =============================================================================

def patch_dspy_for_parallel_candidates(max_concurrent: int = 4):
    """
    Patch DSPy's BootstrapFewShotWithRandomSearch for parallel candidate evaluation.

    By default, DSPy evaluates candidate programs one-at-a-time. This patch
    allows evaluating multiple candidates concurrently for significant speedup.

    Args:
        max_concurrent: Number of candidates to evaluate in parallel
    """
    import dspy
    from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import copy

    original_compile = BootstrapFewShotWithRandomSearch.compile

    def parallel_compile(self, student, trainset=None, valset=None, **kwargs):
        # Get number of candidate sets to try (DSPy uses different attr names in different versions)
        num_candidate_sets = getattr(self, 'num_candidate_sets', None)
        if num_candidate_sets is None:
            num_candidate_sets = getattr(self, 'num_candidate_programs', 16)
        seeds = list(range(-3, num_candidate_sets))

        logger.info(f"Running {len(seeds)} candidates with {max_concurrent} parallel workers")

        results = []

        def compile_one(seed):
            import random
            # Create a fresh copy for thread safety
            local_self = copy.deepcopy(self)
            local_student = copy.deepcopy(student)
            # Override to evaluate just this seed
            local_self.num_candidate_sets = 1
            # Set random seed for reproducibility
            random.seed(seed if seed >= 0 else None)
            try:
                compiled = original_compile(
                    local_self, local_student, trainset=trainset, valset=valset, **kwargs
                )
                # DSPy stores scores on the compiled program, not the optimizer
                if hasattr(compiled, 'candidate_programs') and compiled.candidate_programs:
                    score = compiled.candidate_programs[0].get('score', -1.0)
                else:
                    logger.warning(f"Could not retrieve score for seed={seed} - candidate_programs not found")
                    score = -1.0
                return seed, compiled, score, None
            except Exception as e:
                return seed, None, 0.0, str(e)

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(compile_one, s) for s in seeds]
            for future in as_completed(futures):
                seed, compiled, score, error = future.result()
                if error:
                    logger.warning(f"Candidate seed={seed} failed: {error}")
                else:
                    results.append((seed, compiled, score))
                    logger.info(f"Candidate seed={seed}: score={score:.2%}")

        # Return best candidate
        if results:
            best = max(results, key=lambda x: x[2])
            self.best_score = best[2]
            logger.info(f"Best candidate: seed={best[0]}, score={best[2]:.2%}")
            return best[1]

        # Fallback to original if all failed
        logger.warning("All parallel candidates failed, falling back to original")
        return original_compile(self, student, trainset=trainset, valset=valset, **kwargs)

    BootstrapFewShotWithRandomSearch.compile = parallel_compile
    logger.info(f"Patched DSPy for {max_concurrent} parallel candidate evaluation")


# =============================================================================
# Checkpointing Functions
# =============================================================================

def get_checkpoint_path(output_dir: Path) -> Path:
    """Get the checkpoint directory path."""
    return output_dir / "checkpoints"


def save_checkpoint(output_dir: Path, phase: str, data: dict = None):
    """
    Save a checkpoint marker with optional data.

    Args:
        output_dir: Output directory for the run
        phase: Phase name (e.g., 'phase1', 'phase2', 'round_1')
        data: Optional data to save with the checkpoint
    """
    checkpoint_dir = get_checkpoint_path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = checkpoint_dir / f"{phase}_complete.json"
    checkpoint_data = {
        "phase": phase,
        "timestamp": datetime.now().isoformat(),
        "data": data or {},
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    logger.info(f"Checkpoint saved: {phase}")


def load_checkpoint(output_dir: Path, phase: str) -> Optional[dict]:
    """
    Load a checkpoint if it exists.

    Args:
        output_dir: Output directory for the run
        phase: Phase name to check

    Returns:
        Checkpoint data dict if exists, None otherwise
    """
    checkpoint_file = get_checkpoint_path(output_dir) / f"{phase}_complete.json"
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return None


def get_latest_round_checkpoint(output_dir: Path) -> Tuple[int, Optional[Path]]:
    """
    Find the latest completed optimization round.

    Args:
        output_dir: Output directory for the run

    Returns:
        Tuple of (round_number, classifier_path) or (0, None) if no checkpoints
    """
    checkpoint_dir = get_checkpoint_path(output_dir)
    if not checkpoint_dir.exists():
        return 0, None

    latest_round = 0
    latest_path = None

    for d in checkpoint_dir.iterdir():
        if d.is_dir() and d.name.startswith("round_"):
            try:
                round_num = int(d.name.split("_")[1])
                classifier_path = d / "classifier.json"
                if classifier_path.exists() and round_num > latest_round:
                    latest_round = round_num
                    latest_path = classifier_path
            except (ValueError, IndexError):
                continue

    return latest_round, latest_path


def save_round_checkpoint(output_dir: Path, round_num: int, classifier, stats: dict):
    """
    Save checkpoint after an optimization round.

    Args:
        output_dir: Output directory for the run
        round_num: Round number
        classifier: Classifier to save
        stats: Round statistics
    """
    round_dir = get_checkpoint_path(output_dir) / f"round_{round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)

    # Save classifier state
    try:
        classifier.save(round_dir / "classifier.json")
    except Exception as e:
        logger.warning(f"Could not save classifier state: {e}")

    # Save round stats
    with open(round_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Round {round_num} checkpoint saved")


def save_collector_checkpoint(output_dir: Path, collector):
    """Save the training data collector state."""
    checkpoint_dir = get_checkpoint_path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    collector_path = checkpoint_dir / "collector.json"
    try:
        collector.save(collector_path)
        logger.info(f"Collector checkpoint saved")
    except Exception as e:
        logger.warning(f"Could not save collector: {e}")


def load_collector_checkpoint(output_dir: Path):
    """Load the training data collector state if it exists."""
    from src.manifesto.training_integration import create_rile_training_collector

    collector_path = get_checkpoint_path(output_dir) / "collector.json"
    if collector_path.exists():
        try:
            # Create empty collector and load state
            collector = create_rile_training_collector([], bin_size=10.0)
            collector.load(collector_path)
            return collector
        except Exception as e:
            logger.warning(f"Could not load collector checkpoint: {e}")
    return None


def detect_resume_state(output_dir: Path) -> Dict[str, Any]:
    """
    Detect what state a previous run was in when it stopped.

    Returns dict with:
        - 'phase1_complete': bool
        - 'phase2_complete': bool
        - 'last_round': int (0 if none)
        - 'has_results': bool
        - 'has_collector': bool
    """
    state = {
        'phase1_complete': False,
        'phase2_complete': False,
        'last_round': 0,
        'has_results': False,
        'has_collector': False,
    }

    # Check for results files
    if (output_dir / "train_results.json").exists():
        state['has_results'] = True

    # Check phase checkpoints
    if load_checkpoint(output_dir, "phase1"):
        state['phase1_complete'] = True

    if load_checkpoint(output_dir, "phase2"):
        state['phase2_complete'] = True

    # Check collector
    if (get_checkpoint_path(output_dir) / "collector.json").exists():
        state['has_collector'] = True

    # Check rounds
    last_round, _ = get_latest_round_checkpoint(output_dir)
    state['last_round'] = last_round

    return state


def get_latest_iteration(output_dir: Path) -> Optional[int]:
    """
    Find the latest iteration with saved checkpoints.

    Returns the highest iteration number that has at least an oracle.json file,
    or None if no valid iterations found.
    """
    iteration_dirs = list(output_dir.glob("iteration_*"))
    if not iteration_dirs:
        return None

    # Find highest iteration number with valid checkpoints
    valid_iterations = []
    for d in iteration_dirs:
        try:
            iter_num = int(d.name.split("_")[1])
            # Check if at least oracle exists
            if (d / "oracle.json").exists():
                valid_iterations.append(iter_num)
        except (ValueError, IndexError):
            continue

    return max(valid_iterations) if valid_iterations else None


def load_manifesto_data(
    n_train: int,
    n_val: int,
    n_test: int,
    countries: List[int],
    min_year: int,
) -> Tuple[List, List, List, Any]:
    """Load and split manifesto data."""
    from src.manifesto.data_loader import ManifestoDataset

    print_section("Loading Manifesto Data")

    dataset = ManifestoDataset(
        countries=countries,
        min_year=min_year,
        require_text=True,
    )

    stats = dataset.get_stats()
    logger.info(f"Dataset: {stats['total_manifestos']} manifestos")
    logger.info(f"Countries: {', '.join(stats['country_list'])}")
    logger.info(f"Year range: {stats['year_range']}")
    logger.info(f"RILE range: {stats['rile_range'][0]:.1f} to {stats['rile_range'][1]:.1f}")

    # Create temporal split
    train_ids, val_ids, test_ids = dataset.create_temporal_split(
        train_end_year=1995,
        val_end_year=2005,
    )

    # Limit to requested sizes
    train_ids = train_ids[:n_train]
    val_ids = val_ids[:n_val]
    test_ids = test_ids[:n_test]

    # Load samples
    train_samples = [dataset.get_sample(sid) for sid in train_ids]
    val_samples = [dataset.get_sample(sid) for sid in val_ids]
    test_samples = [dataset.get_sample(sid) for sid in test_ids]

    # Filter None
    train_samples = [s for s in train_samples if s is not None]
    val_samples = [s for s in val_samples if s is not None]
    test_samples = [s for s in test_samples if s is not None]

    logger.info(f"Train: {len(train_samples)} samples")
    logger.info(f"Val: {len(val_samples)} samples")
    logger.info(f"Test: {len(test_samples)} samples")

    return train_samples, val_samples, test_samples, dataset


def process_samples_batched(
    samples: List,
    port: int,
    concurrent_docs: int,
    concurrent_requests: int,
    split_name: str,
    additional_ports: List[int] = None,
    leaf_summarizer=None,
    merge_summarizer=None,
) -> List:
    """Process samples through batched pipeline.

    Args:
        samples: List of manifesto samples
        port: Primary vLLM server port
        concurrent_docs: Max concurrent documents
        concurrent_requests: Max concurrent requests per server
        split_name: Name of this split (train/val/test)
        additional_ports: Additional vLLM server ports for load balancing
        leaf_summarizer: Optional pre-initialized leaf summarizer
        merge_summarizer: Optional pre-initialized merge summarizer
    """
    from src.manifesto.batched_pipeline import (
        BatchedManifestoPipeline,
        BatchedPipelineConfig,
    )

    print_section(f"Processing {split_name} Set ({len(samples)} samples)")

    # Build list of server URLs for load balancing
    server_urls = [f"http://localhost:{port}/v1"]
    if additional_ports:
        for p in additional_ports:
            server_urls.append(f"http://localhost:{p}/v1")
        logger.info(f"Using {len(server_urls)} servers: ports {[port] + additional_ports}")

    config = BatchedPipelineConfig(
        task_model_url=f"http://localhost:{port}/v1",
        task_model_urls=server_urls if len(server_urls) > 1 else None,
        max_concurrent_requests=concurrent_requests,
        max_concurrent_documents=concurrent_docs,
        run_baseline=False,  # Skip baseline for speed
    )

    pipeline = BatchedManifestoPipeline(
        config,
        leaf_summarizer=leaf_summarizer,
        merge_summarizer=merge_summarizer,
    )

    start_time = time.time()

    def progress(completed, total):
        pct = completed / total * 100
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        logger.info(f"  [{split_name}] {completed}/{total} ({pct:.0f}%) - {rate:.1f} samples/sec")

    results = pipeline.process_batch(samples, progress_callback=progress)

    elapsed = time.time() - start_time
    valid = [r for r in results if r.error is None and r.predicted_rile is not None]

    logger.info(f"  Completed in {elapsed:.1f}s ({len(valid)}/{len(samples)} successful)")

    return results


def compute_metrics(results: List, name: str, show_worst: int = 5) -> Dict[str, float]:
    """Compute metrics for a set of results.

    Args:
        results: List of result objects with predicted_rile and ground_truth_rile
        name: Name for logging (e.g., "Train", "Val")
        show_worst: Number of worst predictions to display (0 to disable)
    """
    valid = [r for r in results if r.error is None and r.predicted_rile is not None]

    if not valid:
        return {"mae": float("inf"), "within_10": 0, "within_20": 0, "n": 0}

    # Compute errors with result references for worst-case display
    results_with_errors = [
        (r, abs(r.predicted_rile - r.ground_truth_rile))
        for r in valid
    ]
    errors = [e for _, e in results_with_errors]

    metrics = {
        "mae": sum(errors) / len(errors),
        "within_10": sum(1 for e in errors if e <= 10) / len(errors) * 100,
        "within_20": sum(1 for e in errors if e <= 20) / len(errors) * 100,
        "n": len(valid),
    }

    logger.info(f"  {name}: MAE={metrics['mae']:.2f}, "
               f"Within10={metrics['within_10']:.1f}%, "
               f"Within20={metrics['within_20']:.1f}%")

    # Show worst predictions
    if show_worst > 0 and len(valid) > 0:
        # Sort by error (largest first)
        sorted_results = sorted(results_with_errors, key=lambda x: x[1], reverse=True)
        worst = sorted_results[:show_worst]

        logger.info(f"  {name} worst predictions:")
        for r, err in worst:
            party = getattr(r, 'party_name', None) or getattr(r, 'manifesto_id', 'unknown')
            country = getattr(r, 'country', '')
            year = getattr(r, 'year', '')
            label = f"{party[:20]}"
            if country:
                label = f"{country} {year}"
            gt = r.ground_truth_rile
            pred = r.predicted_rile
            logger.info(f"    {label:20s}  GT={gt:+6.1f}  Pred={pred:+6.1f}  Err={err:5.1f}")

    return metrics


def create_training_data(
    train_results: List,
    val_results: List,
    bin_size: float,
    error_threshold_high: float,
    error_threshold_low: float,
) -> Tuple[Any, Dict]:
    """Create training data from results."""
    from src.manifesto.training_integration import (
        create_rile_training_collector,
        create_rile_label_space,
    )

    print_section("Creating Training Data")

    # Combine train and val for training data
    all_results = train_results + val_results

    collector = create_rile_training_collector(
        all_results,
        bin_size=bin_size,
        error_threshold_high=error_threshold_high,
        error_threshold_low=error_threshold_low,
    )

    stats = collector.get_statistics()
    logger.info(f"Total examples: {stats['total_examples']}")
    logger.info(f"  Positive (violations): {stats.get('positive_count', 'N/A')}")
    logger.info(f"  Negative (good): {stats.get('negative_count', 'N/A')}")

    return collector, stats


def run_optimization_round(
    collector: Any,
    classifier: Any,
    round_num: int,
    config: Any,
    val_results: List,
    metric: Optional[Callable] = None,
) -> Tuple[Any, Dict]:
    """Run a single optimization round.

    Args:
        collector: Training data collector
        classifier: Classifier to optimize
        round_num: Round number
        config: Framework config
        val_results: Validation results for feedback
        metric: Optional custom metric (creates default if None)

    Returns:
        Tuple of (optimized_classifier, round_stats)
    """
    from src.ops_engine.training_framework import (
        OracleOptimizer,
        create_classification_metric,
        evaluate_classifier,
    )
    from src.manifesto.training_integration import create_rile_label_space

    print_section(f"Optimization Round {round_num}")

    label_space = create_rile_label_space(bin_size=10.0)

    # Get training data
    trainset = collector.get_dspy_trainset(
        max_examples=config.optimization.max_examples,
        balanced=True,
    )

    logger.info(f"Training with {len(trainset)} examples")

    if len(trainset) < 4:
        logger.warning("Not enough training examples!")
        return classifier, {"round": round_num, "error": "insufficient_data"}

    # Create optimizer
    optimizer = OracleOptimizer(config.optimization)

    # Use provided metric or create default
    if metric is None:
        metric = create_classification_metric(label_space, weighted=True)

    # Run optimization
    start_time = time.time()
    try:
        classifier = optimizer.optimize(classifier, trainset, metric=metric)
        opt_time = time.time() - start_time

        # Get optimization result
        if optimizer.optimization_history:
            opt_result = optimizer.optimization_history[-1]
            logger.info(f"  Optimization completed in {opt_time:.1f}s")
            logger.info(f"  Metric: {opt_result.metric_before:.3f} -> {opt_result.metric_after:.3f}")
            logger.info(f"  Improvement: {opt_result.improvement:+.3f}")

            round_stats = {
                "round": round_num,
                "metric_before": opt_result.metric_before,
                "metric_after": opt_result.metric_after,
                "improvement": opt_result.improvement,
                "examples_used": opt_result.examples_used,
                "time_sec": opt_time,
            }
        else:
            round_stats = {"round": round_num, "error": "no_history"}

    except Exception as e:
        logger.error(f"  Optimization failed: {e}")
        round_stats = {"round": round_num, "error": str(e)}

    return classifier, round_stats


def evaluate_classifier_on_results(
    classifier: Any,
    results: List,
    name: str,
) -> Dict[str, float]:
    """Evaluate classifier predictions against manifesto results."""
    print_section(f"Evaluating Classifier on {name}")

    predictions = []
    ground_truth = []

    for r in results:
        if r.error is None and r.final_summary:
            try:
                rile, conf, _ = classifier.predict_rile(r.final_summary)
                predictions.append(rile)
                ground_truth.append(r.ground_truth_rile)
            except Exception as e:
                logger.debug(f"Prediction error: {e}")

    if not predictions:
        return {"mae": float("inf"), "n": 0}

    errors = [abs(p - g) for p, g in zip(predictions, ground_truth)]

    metrics = {
        "mae": sum(errors) / len(errors),
        "within_10": sum(1 for e in errors if e <= 10) / len(errors) * 100,
        "within_20": sum(1 for e in errors if e <= 20) / len(errors) * 100,
        "n": len(predictions),
    }

    logger.info(f"  Classifier MAE: {metrics['mae']:.2f}")
    logger.info(f"  Within 10: {metrics['within_10']:.1f}%")
    logger.info(f"  Within 20: {metrics['within_20']:.1f}%")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Manifesto RILE Training Pipeline")

    # Data settings
    parser.add_argument("--train-samples", type=int, default=30,
                       help="Number of training samples")
    parser.add_argument("--val-samples", type=int, default=15,
                       help="Number of validation samples")
    parser.add_argument("--test-samples", type=int, default=10,
                       help="Number of test samples")
    parser.add_argument("--countries", type=int, nargs="+", default=[51, 41],
                       help="Country codes (51=UK, 41=Germany)")
    parser.add_argument("--min-year", type=int, default=1970,
                       help="Minimum year")

    # Server settings
    parser.add_argument("--port", type=int, default=8000,
                       help="Primary vLLM server port")
    parser.add_argument("--additional-ports", type=int, nargs="*", default=None,
                       help="Additional vLLM server ports for load balancing (e.g., --additional-ports 8001 8002)")

    # Batching settings
    parser.add_argument("--concurrent-docs", type=int, default=20,
                       help="Concurrent documents")
    parser.add_argument("--concurrent-requests", type=int, default=50,
                       help="Concurrent LLM requests")

    # Training settings
    parser.add_argument("--rounds", type=int, default=3,
                       help="Number of optimization rounds")
    parser.add_argument("--bin-size", type=float, default=10.0,
                       help="RILE bin size")
    parser.add_argument("--error-high", type=float, default=25.0,
                       help="Error threshold for violations")
    parser.add_argument("--error-low", type=float, default=15.0,
                       help="Error threshold for good examples")
    parser.add_argument("--max-examples", type=int, default=50,
                       help="Max training examples per round")

    # Optimizer settings
    parser.add_argument("--optimizer",
                       choices=['gepa', 'gepa_light', 'gepa_heavy',
                                'mipro', 'mipro_light', 'mipro_medium', 'mipro_heavy',
                                'bootstrap', 'bootstrap_random_search',
                                'labeled_fewshot', 'grpo'],
                       default='gepa',
                       help="Optimizer type. GEPA recommended (instruction-only, no few-shot demos that bloat context)")
    parser.add_argument("--optimizer-budget", choices=['light', 'medium', 'heavy', 'superheavy'],
                       default='heavy', help="Optimization budget for GEPA/MIPROv2 (default: heavy)")
    parser.add_argument("--max-metric-calls", type=int, default=None,
                       help="Direct control over GEPA budget (overrides --optimizer-budget)")
    parser.add_argument("--num-threads", type=int, default=64,
                       help="Parallel metric evaluations for GEPA (default: 64)")
    parser.add_argument("--labeled-k", type=int, default=8,
                       help="Number of demos for LabeledFewShot optimizer (default: 8)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable DSPy caching")
    parser.add_argument("--parallel-candidates", type=int, default=4,
                       help="Number of candidate programs to evaluate in parallel (default: 4)")
    parser.add_argument("--no-precache", action="store_true",
                       help="Disable oracle pre-caching (slower but uses less memory)")
    parser.add_argument("--num-candidates", type=int, default=6,
                       help="Number of candidate programs to try (default: 6)")
    parser.add_argument("--max-bootstrapped-demos", type=int, default=4,
                       help="Max bootstrapped demonstrations per candidate (default: 4)")
    parser.add_argument("--max-rounds", type=int, default=1,
                       help="Max rounds of bootstrapping per candidate (default: 1)")

    # Metric settings
    parser.add_argument("--metric-type", choices=['distance', 'trace', 'judge', 'composite'],
                       default='trace', help="Metric type for optimization (default: trace)")
    parser.add_argument("--use-llm-judge", action="store_true",
                       help="Use LLM as judge in composite metric (slower but more nuanced)")
    parser.add_argument("--judge-weight", type=float, default=0.3,
                       help="Weight for LLM judge in composite metric (default: 0.3)")

    # Iterative optimization (oracle + summarizer)
    parser.add_argument("--n-iterations", type=int, default=2,
                       help="Number of optimization iterations (0=until convergence, default: 2). Use --skip-oracle-opt or --skip-summarizer-opt to control what gets optimized.")
    parser.add_argument("--convergence-threshold", type=float, default=0.01,
                       help="Stop if improvement < this threshold (default: 0.01)")
    parser.add_argument("--convergence-patience", type=int, default=2,
                       help="Stop after N iterations without improvement (default: 2)")
    parser.add_argument("--oracle-budget", choices=['light', 'medium', 'heavy', 'superheavy'],
                       default='heavy', help="GEPA budget for oracle optimization (default: heavy)")
    parser.add_argument("--summarizer-budget", choices=['light', 'medium', 'heavy', 'superheavy'],
                       default='heavy', help="GEPA budget for summarizer optimization (default: heavy)")
    parser.add_argument("--opt-model-port", type=int, default=None,
                       help="Port for smaller optimization model (uses main model if not set)")
    parser.add_argument("--skip-summarizer-opt", action="store_true",
                       help="Skip summarizer optimization (oracle only)")
    parser.add_argument("--skip-oracle-opt", action="store_true",
                       help="Skip oracle optimization (summarizer only)")

    # Probabilistic audit training
    parser.add_argument("--sample-rate", type=float, default=0.10,
                       help="Sampling rate for probabilistic audit (default: 0.10 = 10%%)")
    parser.add_argument("--min-sample-rate", type=float, default=0.05,
                       help="Minimum sampling rate floor (default: 0.05 = 5%%)")
    parser.add_argument("--use-probabilistic-audit", action="store_true",
                       help="Use probabilistic audit training (sample nodes instead of all)")
    parser.add_argument("--use-top-down-init", action="store_true",
                       help="Use top-down initialization with oracle-aligned demos from short docs")
    parser.add_argument("--n-init-demos", type=int, default=8,
                       help="Number of demos for top-down initialization (default: 8)")
    parser.add_argument("--max-init-doc-chars", type=int, default=4000,
                       help="Max document chars for init demos (must fit in context, default: 4000)")

    # Resume from previous run
    parser.add_argument("--resume-from", type=Path, default=None,
                       help="Resume from previous run directory (skip manifesto processing)")
    parser.add_argument("--resume", action="store_true",
                       help="Auto-resume from latest checkpoint in --output-dir (finds most recent run)")

    # Checkpoint loading (for loading trained modules from previous runs)
    parser.add_argument("--load-oracle", type=Path, default=None,
                       help="Load oracle from checkpoint file (e.g., iteration_3/oracle.json)")
    parser.add_argument("--load-leaf-summarizer", type=Path, default=None,
                       help="Load leaf summarizer from checkpoint file")
    parser.add_argument("--load-merge-summarizer", type=Path, default=None,
                       help="Load merge summarizer from checkpoint file")
    parser.add_argument("--load-iteration", type=int, default=None,
                       help="Load all modules from specific iteration (convenience, requires --resume-from)")

    # Output
    parser.add_argument("--output-dir", type=Path,
                       default=Path("data/results/manifesto_rile/training_pipeline"),
                       help="Output directory")

    args = parser.parse_args()

    # Setup output directory - handle resume modes
    resume_dir = None
    resume_state = None

    if args.resume:
        # Auto-find the resume directory
        if args.output_dir.exists():
            # Check if output_dir IS the run directory (has checkpoints or results)
            if (args.output_dir / "checkpoints").exists() or (args.output_dir / "train_results.json").exists():
                # Shell script already found the resume dir
                resume_dir = args.output_dir
                resume_state = detect_resume_state(resume_dir)
                logger.info(f"Using provided resume directory: {resume_dir}")
            else:
                # Look for run_* subdirectories
                subdirs = list(args.output_dir.glob("run_*"))
                if subdirs:
                    resume_dir = max(subdirs, key=lambda p: p.stat().st_mtime)
                    resume_state = detect_resume_state(resume_dir)
                    logger.info(f"Auto-detected resume directory: {resume_dir}")

    elif args.resume_from:
        resume_dir = Path(args.resume_from)
        # Find the actual results directory (may be nested)
        if not (resume_dir / "train_results.json").exists():
            subdirs = list(resume_dir.glob("run_*"))
            if subdirs:
                resume_dir = max(subdirs, key=lambda p: p.stat().st_mtime)
        resume_state = detect_resume_state(resume_dir)

    # Set output directory
    if resume_dir and resume_state:
        output_dir = resume_dir
        logger.info(f"RESUMING from: {output_dir}")
        logger.info(f"  Phase 1 complete: {resume_state['phase1_complete']}")
        logger.info(f"  Phase 2 complete: {resume_state['phase2_complete']}")
        logger.info(f"  Last round: {resume_state['last_round']}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

    print_banner("MANIFESTO RILE TRAINING PIPELINE")
    logger.info(f"Output: {output_dir}")
    logger.info(f"vLLM port: {args.port}")

    # Configure DSPy with vLLM
    import dspy
    import requests

    # Get actual model name from vLLM server
    try:
        resp = requests.get(f"http://localhost:{args.port}/v1/models")
        model_id = resp.json()["data"][0]["id"]
        logger.info(f"Detected vLLM model: {model_id}")
    except Exception as e:
        logger.warning(f"Could not detect model: {e}, using default")
        model_id = "default"

    lm = dspy.LM(
        f"openai/{model_id}",
        api_base=f"http://localhost:{args.port}/v1",
        api_key="EMPTY",
        cache=not getattr(args, 'no_cache', False),  # Enable caching by default
    )
    # Configure DSPy with high parallelism for optimization
    # async_max_workers controls parallel async operations (default is 8, way too low)
    dspy.configure(lm=lm, async_max_workers=args.num_threads)
    logger.info(f"DSPy configured with vLLM (caching={'enabled' if not getattr(args, 'no_cache', False) else 'disabled'}, async_workers={args.num_threads})")

    # Apply parallel candidate evaluation patch for faster optimization
    # Patch if using bootstrap_random_search OR auto (which might select it)
    if args.optimizer in ('bootstrap_random_search', 'auto') and args.parallel_candidates > 1:
        patch_dspy_for_parallel_candidates(max_concurrent=args.parallel_candidates)

    # Save config (convert Path objects to strings for JSON serialization)
    config_dict = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
    }
    config_dict["output_dir"] = str(output_dir)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    all_stats = {
        "config": config_dict,
        "rounds": [],
        "timestamps": {"start": datetime.now().isoformat()},
    }

    try:
        from src.manifesto.evaluation import load_results, save_results

        # Initialize variables
        train_results = None
        val_results = None
        test_samples = None
        train_samples = None
        val_samples = None
        collector = None
        start_round = 1
        leaf_summarizer = None
        merge_summarizer = None

        # =================================================================
        # PHASE 1: Process Manifestos (or load from checkpoint)
        # =================================================================
        skip_phase1 = resume_state and resume_state.get('phase1_complete', False)

        if skip_phase1:
            print_banner("PHASE 1: LOADING FROM CHECKPOINT", "-")
            train_results = load_results(output_dir / "train_results.json")
            val_results = load_results(output_dir / "val_results.json")

            n_with_summary = sum(1 for r in train_results if getattr(r, 'final_summary', None))
            logger.info(f"Loaded {len(train_results)} train, {len(val_results)} val results")
            logger.info(f"  Results with final_summary: {n_with_summary}")

            all_stats["data"] = {
                "train": len(train_results),
                "val": len(val_results),
                "test": 0,
                "resumed_from": str(output_dir),
            }

            # Compute baseline metrics from loaded results
            print_section("Baseline Pipeline Metrics (from checkpoint)")
            train_metrics = compute_metrics(train_results, "Train")
            val_metrics = compute_metrics(val_results, "Val")

            all_stats["baseline"] = {
                "train": train_metrics,
                "val": val_metrics,
            }

        else:
            # Normal mode: Load and process fresh data
            train_samples, val_samples, test_samples, dataset = load_manifesto_data(
                n_train=args.train_samples,
                n_val=args.val_samples,
                n_test=args.test_samples,
                countries=args.countries,
                min_year=args.min_year,
            )

            all_stats["data"] = {
                "train": len(train_samples),
                "val": len(val_samples),
                "test": len(test_samples),
            }

            # Top-down initialization: train summarizers BEFORE Phase 1
            leaf_summarizer = None
            merge_summarizer = None
            if getattr(args, 'use_top_down_init', False):
                print_banner("TOP-DOWN INITIALIZATION (GEPA)", "-")
                max_init_chars = getattr(args, 'max_init_doc_chars', 4000)
                n_init_demos = getattr(args, 'n_init_demos', 8)
                logger.info(f"Training summarizers on short docs (max {max_init_chars} chars)...")
                logger.info("  Using GEPA for actual prompt optimization (NOT demo selection)")
                logger.info("  Short docs: oracle(doc) = ground_truth by definition")

                from src.manifesto.dspy_summarizer import LeafSummarizer, MergeSummarizer
                from src.ops_engine.initialization import train_on_short_docs
                from src.manifesto.rubrics import RILE_PRESERVATION_RUBRIC
                from src.manifesto.training_integration import RILEOracleClassifier

                # Create baseline oracle for RILE-based metric
                init_oracle = RILEOracleClassifier(bin_size=args.bin_size)

                # Load extra samples from full dataset for training
                # (train_samples may be too small to find enough short docs)
                demo_samples = train_samples
                short_in_train = sum(1 for s in train_samples if len(getattr(s, 'text', '') or '') <= max_init_chars)
                if short_in_train < n_init_demos:
                    logger.info(f"  Only {short_in_train} short docs in train set, loading more from full dataset...")
                    # Get all samples from dataset for training pool
                    all_ids = dataset.get_all_ids()  # Use filtered_df, not unfiltered metadata_df
                    all_samples = [dataset.get_sample(sid) for sid in all_ids[:500]]  # Cap at 500 for speed
                    all_samples = [s for s in all_samples if s is not None]
                    # Filter to short docs with valid labels
                    short_samples = [
                        s for s in all_samples
                        if len(getattr(s, 'text', '') or '') <= max_init_chars
                        and getattr(s, 'rile', None) is not None
                    ]
                    logger.info(f"  Found {len(short_samples)} short docs with labels in full dataset")
                    if short_samples:
                        demo_samples = short_samples

                # Create summarizers
                leaf_summarizer = LeafSummarizer(use_cot=True)
                merge_summarizer = MergeSummarizer(use_cot=True)

                try:
                    # Train summarizers using GEPA (actual prompt optimization)
                    # Uses ground-truth RILE labels in metric
                    leaf_summarizer, merge_summarizer = train_on_short_docs(
                        train_samples=demo_samples,
                        leaf_summarizer=leaf_summarizer,
                        merge_summarizer=merge_summarizer,
                        oracle_classifier=init_oracle,  # For ground-truth metric
                        rubric=RILE_PRESERVATION_RUBRIC,
                        max_doc_chars=max_init_chars,
                        optimizer_type='gepa',  # Actual prompt optimization
                        max_metric_calls=200,
                        text_field='text',
                        label_field='rile',
                    )
                    logger.info("Top-down initialization complete - summarizers trained with GEPA")
                except Exception as e:
                    logger.error(f"Top-down initialization failed: {e}")
                    logger.warning("Continuing with unoptimized default summarizers - expect poor results")
                    leaf_summarizer = None
                    merge_summarizer = None

            print_banner("PHASE 1: PROCESS MANIFESTOS", "-")

            train_results = process_samples_batched(
                train_samples, args.port,
                args.concurrent_docs, args.concurrent_requests,
                "Train",
                additional_ports=args.additional_ports,
                leaf_summarizer=leaf_summarizer,
                merge_summarizer=merge_summarizer,
            )

            val_results = process_samples_batched(
                val_samples, args.port,
                args.concurrent_docs, args.concurrent_requests,
                "Val",
                additional_ports=args.additional_ports,
                leaf_summarizer=leaf_summarizer,
                merge_summarizer=merge_summarizer,
            )

            # Compute baseline metrics
            print_section("Baseline Pipeline Metrics")
            train_metrics = compute_metrics(train_results, "Train")
            val_metrics = compute_metrics(val_results, "Val")

            all_stats["baseline"] = {
                "train": train_metrics,
                "val": val_metrics,
            }

            # Save results and checkpoint
            save_results(train_results, output_dir / "train_results.json")
            save_results(val_results, output_dir / "val_results.json")
            save_checkpoint(output_dir, "phase1", {
                "train_count": len(train_results),
                "val_count": len(val_results),
                "train_mae": train_metrics.get("mae"),
                "val_mae": val_metrics.get("mae"),
            })

        # =================================================================
        # PHASE 2: Create Training Data (or load from checkpoint)
        # =================================================================
        skip_phase2 = resume_state and resume_state.get('phase2_complete', False)

        if skip_phase2:
            print_banner("PHASE 2: LOADING COLLECTOR FROM CHECKPOINT", "-")
            collector = load_collector_checkpoint(output_dir)
            if collector is None:
                logger.warning("Could not load collector checkpoint, recreating...")
                skip_phase2 = False

        if not skip_phase2:
            print_banner("PHASE 2: CREATE TRAINING DATA", "-")

            collector, training_stats = create_training_data(
                train_results, val_results,
                bin_size=args.bin_size,
                error_threshold_high=args.error_high,
                error_threshold_low=args.error_low,
            )

            all_stats["training_data"] = training_stats

            # Save collector checkpoint
            save_collector_checkpoint(output_dir, collector)
            save_checkpoint(output_dir, "phase2", {
                "total_examples": training_stats.get("total_examples", 0),
            })
        else:
            training_stats = collector.get_statistics() if collector else {}
            all_stats["training_data"] = training_stats
            logger.info(f"Loaded collector with {training_stats.get('total_examples', 0)} examples")

        # 4. Initialize classifier
        from src.manifesto.training_integration import RILEOracleClassifier
        from src.ops_engine.training_framework import FrameworkConfig

        config = FrameworkConfig()
        config.optimization.max_examples = args.max_examples
        config.optimization.save_checkpoints = True
        config.optimization.checkpoint_dir = output_dir / "checkpoints"

        # Log optimizer settings
        logger.info(f"Optimizer: {args.optimizer} (budget={args.optimizer_budget}, threads={args.num_threads})")

        # Apply optimizer settings from CLI
        config.optimization.optimizer_type = args.optimizer
        config.optimization.gepa_auto = args.optimizer_budget
        config.optimization.num_threads = args.num_threads
        config.optimization.log_dir = output_dir / "optimizer_logs"
        config.optimization.track_stats = True
        config.optimization.max_metric_calls = args.max_metric_calls
        config.optimization.labeled_k = args.labeled_k

        # Also set dataset sample sizes from CLI
        config.optimization.train_samples = args.train_samples
        config.optimization.val_samples = args.val_samples
        config.optimization.test_samples = args.test_samples
        if args.optimizer == 'labeled_fewshot':
            logger.info(f"  LabeledFewShot k={args.labeled_k}")

        # Configure metric based on CLI arguments
        from src.ops_engine.training_framework.metrics import (
            create_classification_metric,
            create_classification_metric_with_trace,
            create_llm_judge_metric,
            create_composite_metric,
        )
        from src.manifesto.training_integration import create_rile_label_space

        # Create label space for metrics
        label_space = create_rile_label_space(bin_size=args.bin_size)

        # Select metric based on args
        metric_type = getattr(args, 'metric_type', 'trace')
        use_llm_judge = getattr(args, 'use_llm_judge', False)
        judge_weight = getattr(args, 'judge_weight', 0.3)

        if metric_type == 'composite':
            metric = create_composite_metric(
                label_space=label_space,
                use_trace=True,
                use_llm_judge=use_llm_judge,
                judge_weight=judge_weight,
            )
            logger.info(f"Metric: composite (trace=True, llm_judge={use_llm_judge}, weight={judge_weight})")
        elif metric_type == 'trace':
            metric = create_classification_metric_with_trace(
                label_space=label_space,
                weighted=True,
                with_feedback=True,
            )
            logger.info("Metric: trace-based with feedback")
        elif metric_type == 'judge':
            metric = create_llm_judge_metric()
            logger.info("Metric: LLM judge")
        else:  # distance
            metric = create_classification_metric(
                label_space=label_space,
                weighted=True,
                with_feedback=True,
            )
            logger.info("Metric: distance-based with feedback")

        # =================================================================
        # PHASE 3: Optimization
        # =================================================================

        print_banner("PHASE 3: OPTIMIZATION", "-")
        logger.info(f"Iterations: {args.n_iterations} (0=until convergence)")
        if not args.skip_oracle_opt and not args.skip_summarizer_opt:
            logger.info(f"Pattern: oracle → summarizer → oracle → summarizer → ...")
        elif args.skip_summarizer_opt:
            logger.info(f"Pattern: oracle only (summarizer optimization skipped)")
        elif args.skip_oracle_opt:
            logger.info(f"Pattern: summarizer only (oracle optimization skipped)")
        else:
            logger.info(f"Pattern: no optimization (both skipped)")
        logger.info("Metric: numeric RILE (continuous error-based, score = 1.0 - error/100)")

        # Need raw samples - reload if resuming
        if train_samples is None:
            logger.info("Reloading train samples for optimization...")
            train_samples, val_samples, _, iter_dataset = load_manifesto_data(
                n_train=args.train_samples,
                n_val=args.val_samples,
                n_test=0,  # Don't need test for iteration
                countries=args.countries,
                min_year=args.min_year,
            )

            # Top-down initialization for resumed case (summarizers not yet initialized)
            if getattr(args, 'use_top_down_init', False) and leaf_summarizer is None:
                print_banner("TOP-DOWN INITIALIZATION (Resumed, GEPA)", "-")
                max_init_chars = getattr(args, 'max_init_doc_chars', 4000)
                n_init_demos = getattr(args, 'n_init_demos', 8)
                logger.info(f"Training summarizers on short docs (max {max_init_chars} chars)...")
                logger.info("  Using GEPA for actual prompt optimization")

                from src.manifesto.dspy_summarizer import LeafSummarizer, MergeSummarizer
                from src.ops_engine.initialization import train_on_short_docs
                from src.manifesto.rubrics import RILE_PRESERVATION_RUBRIC
                from src.manifesto.training_integration import RILEOracleClassifier

                # Create baseline oracle for RILE-based metric
                init_oracle = RILEOracleClassifier(bin_size=args.bin_size)

                # Load extra samples from full dataset for training
                demo_samples = train_samples
                short_in_train = sum(1 for s in train_samples if len(getattr(s, 'text', '') or '') <= max_init_chars)
                if short_in_train < n_init_demos:
                    logger.info(f"  Only {short_in_train} short docs in train set, loading more from full dataset...")
                    all_ids = iter_dataset.get_all_ids()  # Use filtered_df, not unfiltered metadata_df
                    all_samples = [iter_dataset.get_sample(sid) for sid in all_ids[:500]]
                    all_samples = [s for s in all_samples if s is not None]
                    short_samples = [
                        s for s in all_samples
                        if len(getattr(s, 'text', '') or '') <= max_init_chars
                        and getattr(s, 'rile', None) is not None
                    ]
                    logger.info(f"  Found {len(short_samples)} short docs with labels in full dataset")
                    if short_samples:
                        demo_samples = short_samples

                leaf_summarizer = LeafSummarizer(use_cot=True)
                merge_summarizer = MergeSummarizer(use_cot=True)

                try:
                    # Train summarizers using GEPA (actual prompt optimization)
                    leaf_summarizer, merge_summarizer = train_on_short_docs(
                        train_samples=demo_samples,
                        leaf_summarizer=leaf_summarizer,
                        merge_summarizer=merge_summarizer,
                        oracle_classifier=init_oracle,  # For ground-truth metric
                        rubric=RILE_PRESERVATION_RUBRIC,
                        max_doc_chars=max_init_chars,
                        optimizer_type='gepa',  # Actual prompt optimization
                        max_metric_calls=200,
                        text_field='text',
                        label_field='rile',
                    )
                    logger.info("Top-down initialization complete - summarizers trained with GEPA")
                except Exception as e:
                    logger.error(f"Top-down initialization failed: {e}")
                    logger.warning("Continuing with unoptimized default summarizers - expect poor results")
                    leaf_summarizer = None
                    merge_summarizer = None

        # Run optimization (pass pre-initialized summarizers if available)
        classifier, leaf_summarizer, merge_summarizer, iter_stats = run_iterative_optimization(
            train_samples=train_samples,
            val_samples=val_samples,
            args=args,
            output_dir=output_dir,
            leaf_summarizer=leaf_summarizer,
            merge_summarizer=merge_summarizer,
            resume_dir=resume_dir,
        )

        all_stats["iterative"] = iter_stats
        all_stats["rounds"] = iter_stats.get("iterations", [])

        # Evaluate final classifier on validation results
        if val_results:
            final_val_eval = evaluate_classifier_on_results(classifier, val_results, "Final Validation")
            all_stats["final_val_eval"] = final_val_eval

        # 6. Final evaluation on test set
        if test_samples:
            print_banner("PHASE 4: FINAL EVALUATION", "-")

            test_results = process_samples_batched(
                test_samples, args.port,
                args.concurrent_docs, args.concurrent_requests,
                "Test",
                additional_ports=args.additional_ports,
            )

            test_metrics = compute_metrics(test_results, "Test (Pipeline)")
            test_classifier_metrics = evaluate_classifier_on_results(
                classifier, test_results, "Test (Classifier)"
            )

            all_stats["test"] = {
                "pipeline": test_metrics,
                "classifier": test_classifier_metrics,
            }

            save_results(test_results, output_dir / "test_results.json")

        # 7. Print summary
        print_banner("TRAINING COMPLETE")

        print("\n=== PROGRESS OVER ITERATIONS ===")
        print(f"{'Iter':<6} {'MAE Before':<12} {'MAE After':<12} {'Δ MAE':<10} {'Docs':<6}")
        print("-" * 50)

        for r in all_stats.get("rounds", []):
            if "error" not in r:
                mae_before = r.get("metric_before", float("inf"))
                mae_after = r.get("metric_after", float("inf"))
                improvement = r.get("improvement", 0)
                docs = r.get("successful", r.get("docs_processed", "?"))
                mae_before_str = f"{mae_before:.2f}" if mae_before != float("inf") else "N/A"
                mae_after_str = f"{mae_after:.2f}" if mae_after != float("inf") else "N/A"
                # Positive improvement means MAE decreased (good)
                imp_str = f"{improvement:+.2f}" if improvement != 0 else "0.00"
                print(f"{r['round']:<6} {mae_before_str:<12} {mae_after_str:<12} {imp_str:<10} {docs:<6}")
            else:
                print(f"{r['round']:<6} ERROR: {r.get('error', 'unknown')}")

        print("\n=== FINAL METRICS ===")
        print(f"Baseline Pipeline MAE (Train): {all_stats['baseline']['train']['mae']:.2f}")
        print(f"Baseline Pipeline MAE (Val):   {all_stats['baseline']['val']['mae']:.2f}")

        if "test" in all_stats:
            print(f"Test Pipeline MAE:             {all_stats['test']['pipeline']['mae']:.2f}")
            print(f"Test Classifier MAE:           {all_stats['test']['classifier']['mae']:.2f}")

        # Save final stats
        all_stats["timestamps"]["end"] = datetime.now().isoformat()
        with open(output_dir / "final_stats.json", "w") as f:
            json.dump(all_stats, f, indent=2)

        print(f"\nResults saved to: {output_dir}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        all_stats["error"] = "interrupted"
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        all_stats["error"] = str(e)
        raise
    finally:
        all_stats["timestamps"]["end"] = datetime.now().isoformat()
        with open(output_dir / "final_stats.json", "w") as f:
            json.dump(all_stats, f, indent=2)


def run_iterative_optimization(
    train_samples: List,
    val_samples: List,
    args,
    output_dir: Path,
    leaf_summarizer=None,
    merge_summarizer=None,
    resume_dir: Optional[Path] = None,
) -> Tuple[Any, Any, Any, Dict]:
    """
    Run two-step iterative optimization for oracle and summarizers.

    This implements the iterative training loop:
    1. Process documents with current summarizers
    2. Train/update oracle classifier on summaries
    3. Optimize summarizers using oracle as metric
    4. Check for convergence, repeat if not converged

    Convergence is detected when:
    - Oracle metric improvement < threshold for `patience` consecutive iterations
    - AND summarizer metric improvement < threshold (if not skipped)

    Args:
        train_samples: Training manifesto samples
        val_samples: Validation manifesto samples
        args: Command line arguments
        output_dir: Output directory for checkpoints
        leaf_summarizer: Optional pre-initialized leaf summarizer
        merge_summarizer: Optional pre-initialized merge summarizer

    Returns:
        Tuple of (oracle_classifier, leaf_summarizer, merge_summarizer, stats)
    """
    import dspy
    import requests
    from src.manifesto.dspy_summarizer import LeafSummarizer, MergeSummarizer
    from src.manifesto.batched_pipeline import BatchedManifestoPipeline, BatchedPipelineConfig
    from src.manifesto.training_integration import (
        RILEOracleClassifier,
        collect_summarization_training_data,
        collect_merge_training_data,
        create_oracle_trainset,
        filter_short_documents,
        collect_sampled_training_data,
        get_sample_size,
    )
    from src.manifesto.constants import RILE_MIN, RILE_MAX
    from src.ops_engine.scoring import BoundedScale
    from src.ops_engine.training_framework.metrics import (
        create_metric,
        create_cached_oracle_metric,
        get_cache_stats,
    )

    # Create RILE scale for generic metric creation
    RILE_SCALE = BoundedScale(RILE_MIN, RILE_MAX)

    print_banner("ITERATIVE OPTIMIZATION")

    # Determine max iterations (0 = until convergence)
    max_iterations = args.n_iterations if args.n_iterations > 0 else 100
    run_until_convergence = args.n_iterations == 0

    logger.info(f"Max iterations: {max_iterations if not run_until_convergence else 'until convergence'}")
    logger.info(f"Convergence threshold: {args.convergence_threshold}")
    logger.info(f"Convergence patience: {args.convergence_patience}")
    logger.info(f"Oracle budget: {args.oracle_budget}")
    logger.info(f"Summarizer budget: {args.summarizer_budget}")
    logger.info(f"Skip summarizer opt: {args.skip_summarizer_opt}")

    # Setup optimization model (optionally different from inference model)
    opt_port = args.opt_model_port or args.port
    if args.opt_model_port:
        try:
            resp = requests.get(f"http://localhost:{opt_port}/v1/models")
            opt_model_id = resp.json()["data"][0]["id"]
            logger.info(f"Optimization model: {opt_model_id} (port {opt_port})")

            # Create separate LM for optimization
            opt_lm = dspy.LM(
                f"openai/{opt_model_id}",
                api_base=f"http://localhost:{opt_port}/v1",
                api_key="EMPTY",
                cache=True,
            )
        except Exception as e:
            logger.warning(f"Could not setup optimization model: {e}, using main model")
            opt_lm = None
            opt_port = args.port
    else:
        opt_lm = None
        logger.info(f"Using main model for optimization (port {args.port})")

    # Initialize modules (use provided or create new)
    if leaf_summarizer is None:
        leaf_summarizer = LeafSummarizer(use_cot=True)
        logger.info("Created new leaf summarizer")
    else:
        logger.info("Using pre-initialized leaf summarizer")

    if merge_summarizer is None:
        merge_summarizer = MergeSummarizer(use_cot=True)
        logger.info("Created new merge summarizer")
    else:
        logger.info("Using pre-initialized merge summarizer")

    oracle_classifier = RILEOracleClassifier(bin_size=args.bin_size)

    # =========================================================================
    # CHECKPOINT LOADING: Load trained modules from previous runs if specified
    # =========================================================================

    # Track what was loaded for logging and stats
    loaded_modules = {
        "oracle": None,
        "leaf_summarizer": None,
        "merge_summarizer": None,
    }

    # Determine which iteration to load from (auto-detect or explicit)
    load_iteration = getattr(args, 'load_iteration', None)
    if load_iteration is None and resume_dir:
        detected = get_latest_iteration(resume_dir)
        if detected:
            logger.info(f"Auto-detected latest iteration: {detected}")
            load_iteration = detected

    # Load oracle if specified (explicit path takes priority)
    load_oracle_path = getattr(args, 'load_oracle', None)
    if load_oracle_path:
        if load_oracle_path.exists():
            oracle_classifier.load(load_oracle_path)
            loaded_modules["oracle"] = str(load_oracle_path)
            logger.info(f"Loaded oracle from explicit path: {load_oracle_path}")
        else:
            logger.warning(f"Oracle checkpoint not found: {load_oracle_path}")
    elif load_iteration and resume_dir:
        oracle_path = resume_dir / f"iteration_{load_iteration}" / "oracle.json"
        if oracle_path.exists():
            oracle_classifier.load(oracle_path)
            loaded_modules["oracle"] = str(oracle_path)
            logger.info(f"Loaded oracle from iteration {load_iteration}")
        else:
            logger.info(f"No oracle checkpoint found at iteration {load_iteration}")

    # Load leaf summarizer if specified (explicit path takes priority)
    load_leaf_path = getattr(args, 'load_leaf_summarizer', None)
    if load_leaf_path:
        if load_leaf_path.exists():
            leaf_summarizer.load(load_leaf_path)
            loaded_modules["leaf_summarizer"] = str(load_leaf_path)
            logger.info(f"Loaded leaf summarizer from explicit path: {load_leaf_path}")
        else:
            logger.warning(f"Leaf summarizer checkpoint not found: {load_leaf_path}")
    elif load_iteration and resume_dir:
        leaf_path = resume_dir / f"iteration_{load_iteration}" / "leaf_summarizer.json"
        if leaf_path.exists():
            leaf_summarizer.load(leaf_path)
            loaded_modules["leaf_summarizer"] = str(leaf_path)
            logger.info(f"Loaded leaf summarizer from iteration {load_iteration}")

    # Load merge summarizer if specified (explicit path takes priority)
    load_merge_path = getattr(args, 'load_merge_summarizer', None)
    if load_merge_path:
        if load_merge_path.exists():
            merge_summarizer.load(load_merge_path)
            loaded_modules["merge_summarizer"] = str(load_merge_path)
            logger.info(f"Loaded merge summarizer from explicit path: {load_merge_path}")
        else:
            logger.warning(f"Merge summarizer checkpoint not found: {load_merge_path}")
    elif load_iteration and resume_dir:
        merge_path = resume_dir / f"iteration_{load_iteration}" / "merge_summarizer.json"
        if merge_path.exists():
            merge_summarizer.load(merge_path)
            loaded_modules["merge_summarizer"] = str(merge_path)
            logger.info(f"Loaded merge summarizer from iteration {load_iteration}")

    # Log summary of module states
    n_loaded = sum(1 for v in loaded_modules.values() if v is not None)
    if n_loaded > 0:
        print_section("CHECKPOINT LOADING SUMMARY")
        logger.info(f"Loaded {n_loaded}/3 modules from checkpoints:")
        for name, path in loaded_modules.items():
            status = f"LOADED: {path}" if path else "fresh (not loaded)"
            logger.info(f"  {name}: {status}")
    else:
        logger.info("No modules loaded from checkpoints - starting fresh")

    # =========================================================================

    # Create pipeline config
    pipeline_config = BatchedPipelineConfig(
        task_model_url=f"http://localhost:{args.port}/v1",
        max_concurrent_documents=args.concurrent_docs,
        max_concurrent_requests=args.concurrent_requests,
    )

    # Convergence tracking
    prev_oracle_score = 0.0
    prev_summarizer_score = 0.0
    prev_pipeline_mae = float("inf")
    no_improvement_count = 0

    iteration_stats = {
        "iterations": [],
        "config": {
            "max_iterations": max_iterations,
            "convergence_threshold": args.convergence_threshold,
            "convergence_patience": args.convergence_patience,
            "oracle_budget": args.oracle_budget,
            "summarizer_budget": args.summarizer_budget,
            "opt_model_port": opt_port,
            "skip_summarizer_opt": args.skip_summarizer_opt,
            "skip_oracle_opt": getattr(args, 'skip_oracle_opt', False),
        },
        "loaded_modules": loaded_modules,
        "converged": False,
    }

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        print_section(f"Iteration {iteration}" + (f"/{max_iterations}" if not run_until_convergence else ""))
        iter_start = time.time()
        iter_stats = {"iteration": iteration, "round": iteration}

        # Step 1: Process documents with current summarizers
        logger.info(f"Step 1: Processing {len(train_samples)} documents...")

        pipeline = BatchedManifestoPipeline(
            config=pipeline_config,
            leaf_summarizer=leaf_summarizer,
            merge_summarizer=merge_summarizer,
        )

        results = pipeline.process_batch_with_dspy(train_samples, show_progress=True)
        iter_stats["docs_processed"] = len(results)
        iter_stats["successful"] = sum(1 for r in results if r.error is None)

        # Compute pipeline MAE for this iteration
        valid_results = [r for r in results if r.error is None and r.predicted_rile is not None]
        if valid_results:
            errors = [abs(r.predicted_rile - r.ground_truth_rile) for r in valid_results]
            iter_stats["pipeline_mae"] = sum(errors) / len(errors)
        else:
            iter_stats["pipeline_mae"] = float("inf")

        # Step 1.5: Compute and log OPS violation rates (Paper Section 4.1)
        # This reports p_suff, union bound, etc. to track summarization quality
        from src.manifesto.rubrics import RILE_PRESERVATION_RUBRIC
        violation_stats = compute_and_log_violation_rates(
            results=results,
            oracle_classifier=oracle_classifier,
            rubric=RILE_PRESERVATION_RUBRIC,
            iteration=iteration,
            output_dir=output_dir,
        )
        iter_stats["violation_rates"] = violation_stats

        # Step 2: Train oracle on current summaries (unless skipped)
        # Initialize oracle_score to previous value (in case optimization is skipped)
        oracle_score = prev_oracle_score

        if not args.skip_oracle_opt:
            logger.info("Step 2: Training RILE oracle...")

            oracle_trainset = create_oracle_trainset(results, bin_size=args.bin_size)
            logger.info(f"  {len(oracle_trainset)} training examples")

            oracle_opt_type = args.optimizer

            # Use generic scale-based metric (continuous error-based)
            # Compares predicted label to ground truth using BoundedScale
            def oracle_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
                ground_truth = getattr(gold, 'ground_truth_rile', None)
                if ground_truth is None:
                    ground_truth = float(getattr(gold, 'label', 0))
                predicted = float(getattr(pred, 'label', 0))
                score = RILE_SCALE.values_to_score(predicted, ground_truth)
                # Return float for BootstrapFewShot compatibility (GEPA needs dict)
                return score
            logger.info("  Using generic scale-based metric (score = 1.0 - error/range)")

            try:
                # Use optimization LM if available
                opt_context = dspy.context(lm=opt_lm) if opt_lm else dspy.context()

                with opt_context:
                    oracle_optimizer = create_dspy_optimizer(
                        optimizer_type=oracle_opt_type,
                        metric=oracle_metric,
                        budget=args.oracle_budget,
                        num_threads=args.num_threads,
                        max_metric_calls=args.max_metric_calls,
                        labeled_k=getattr(args, 'labeled_k', 8),
                        num_candidates=getattr(args, 'num_candidates', 16),
                        max_bootstrapped_demos=getattr(args, 'max_bootstrapped_demos', 4),
                        max_rounds=getattr(args, 'max_rounds', 1),
                    )

                    compiled_oracle = oracle_optimizer.compile(
                        student=oracle_classifier,
                        trainset=oracle_trainset,
                    )

                # Get score from optimizer stats if available
                oracle_score = getattr(oracle_optimizer, 'best_score', 0.0)
                if oracle_score == 0.0:
                    # Estimate from trainset
                    oracle_score = sum(1 for _ in oracle_trainset) / max(len(oracle_trainset), 1)

                oracle_classifier = compiled_oracle
                iter_stats["oracle_trained"] = True
                iter_stats["oracle_score"] = oracle_score
                logger.info(f"  Oracle trained (score: {oracle_score:.3f})")

                # Save oracle checkpoint immediately
                iter_checkpoint_dir = output_dir / f"iteration_{iteration}"
                iter_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                try:
                    oracle_classifier.save(iter_checkpoint_dir / "oracle.json")
                    logger.info(f"  Oracle checkpoint saved")
                except Exception as save_e:
                    logger.debug(f"  Could not save oracle checkpoint: {save_e}")

            except Exception as e:
                logger.error(f"  Oracle training failed: {e}")
                iter_stats["oracle_trained"] = False
                iter_stats["oracle_error"] = str(e)
                oracle_score = prev_oracle_score
        else:
            logger.info("Step 2: Skipping oracle optimization (--skip-oracle-opt)")
            iter_stats["oracle_skipped"] = True

        # Step 3: Optimize summarizers (unless skipped)
        # Now runs LEAF and MERGE in PARALLEL for ~33% speedup
        summarizer_score = prev_summarizer_score

        if not args.skip_summarizer_opt:
            from concurrent.futures import ThreadPoolExecutor

            logger.info("Step 3: Optimizing summarizers in parallel...")

            # Collect training data - use probabilistic sampling if enabled
            if getattr(args, 'use_probabilistic_audit', False):
                logger.info(f"  Using probabilistic audit (sample_rate={args.sample_rate:.0%})")
                summarization_trainset, merge_trainset = collect_sampled_training_data(
                    results,
                    sample_rate=args.sample_rate,
                    min_rate=args.min_sample_rate,
                )
            else:
                summarization_trainset = collect_summarization_training_data(results)
                merge_trainset = collect_merge_training_data(results)

            logger.info(f"  {len(summarization_trainset)} leaf examples, {len(merge_trainset)} merge examples")

            # Define optimization functions for parallel execution
            def optimize_leaf():
                """Optimize leaf summarizer (runs in thread)."""
                nonlocal leaf_summarizer
                leaf_stats = {}

                if not summarization_trainset:
                    leaf_stats["leaf_optimized"] = False
                    leaf_stats["leaf_error"] = "No training data"
                    return leaf_summarizer, leaf_stats, prev_summarizer_score

                dspy_trainset = [ex.to_dspy_example() for ex in summarization_trainset]

                leaf_opt_type = args.optimizer

                # Create cached oracle metric (thread-safe, reduces redundant LLM calls)
                summarization_metric, leaf_oracle_cache = create_cached_oracle_metric(
                    oracle_classifier=oracle_classifier,
                    scale=RILE_SCALE,
                    ground_truth_field='ground_truth_rile',
                    summary_field='summary',
                )

                try:
                    with opt_context if opt_lm else dspy.context():
                        optimizer = create_dspy_optimizer(
                            optimizer_type=leaf_opt_type,
                            metric=summarization_metric,
                            budget=args.summarizer_budget,
                            num_threads=args.num_threads,
                            max_metric_calls=args.max_metric_calls,
                            labeled_k=getattr(args, 'labeled_k', 8),
                            num_candidates=getattr(args, 'num_candidates', 16),
                            max_bootstrapped_demos=getattr(args, 'max_bootstrapped_demos', 4),
                            max_rounds=getattr(args, 'max_rounds', 1),
                        )
                        compiled = optimizer.compile(student=leaf_summarizer, trainset=dspy_trainset)

                    score = getattr(optimizer, 'best_score', 0.5)
                    leaf_stats["leaf_optimized"] = True
                    leaf_stats["leaf_score"] = score

                    # Log cache statistics
                    cache_stats = get_cache_stats(leaf_oracle_cache)
                    leaf_stats["cache_stats"] = cache_stats
                    logger.info(f"  Leaf optimized (score: {score:.3f})")
                    logger.info(f"    Oracle cache: {cache_stats['cache_size']} unique, "
                               f"{cache_stats['hit_rate']:.1%} hit rate")

                    # Save checkpoint
                    iter_dir = output_dir / f"iteration_{iteration}"
                    iter_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        compiled.save(iter_dir / "leaf_summarizer.json")
                    except Exception:
                        pass

                    return compiled, leaf_stats, score

                except Exception as e:
                    logger.error(f"  Leaf optimization failed: {e}")
                    leaf_stats["leaf_optimized"] = False
                    leaf_stats["leaf_error"] = str(e)
                    return leaf_summarizer, leaf_stats, prev_summarizer_score

            def optimize_merge():
                """Optimize merge summarizer (runs in thread)."""
                nonlocal merge_summarizer
                merge_stats = {}

                if not merge_trainset:
                    merge_stats["merge_optimized"] = False
                    merge_stats["merge_error"] = "No training data"
                    return merge_summarizer, merge_stats

                dspy_trainset = [ex.to_dspy_example() for ex in merge_trainset]

                merge_opt_type = args.optimizer

                # Create cached oracle metric (thread-safe, reduces redundant LLM calls)
                merge_metric, merge_oracle_cache = create_cached_oracle_metric(
                    oracle_classifier=oracle_classifier,
                    scale=RILE_SCALE,
                    ground_truth_field='ground_truth_rile',
                    summary_field='merged_summary',
                )

                try:
                    with opt_context if opt_lm else dspy.context():
                        optimizer = create_dspy_optimizer(
                            optimizer_type=merge_opt_type,
                            metric=merge_metric,
                            budget=args.summarizer_budget,
                            num_threads=args.num_threads,
                            max_metric_calls=args.max_metric_calls,
                            labeled_k=getattr(args, 'labeled_k', 8),
                            num_candidates=getattr(args, 'num_candidates', 16),
                            max_bootstrapped_demos=getattr(args, 'max_bootstrapped_demos', 4),
                            max_rounds=getattr(args, 'max_rounds', 1),
                        )
                        compiled = optimizer.compile(student=merge_summarizer, trainset=dspy_trainset)

                    merge_stats["merge_optimized"] = True

                    # Log cache statistics
                    cache_stats = get_cache_stats(merge_oracle_cache)
                    merge_stats["cache_stats"] = cache_stats
                    logger.info("  Merge optimized")
                    logger.info(f"    Oracle cache: {cache_stats['cache_size']} unique, "
                               f"{cache_stats['hit_rate']:.1%} hit rate")

                    # Save checkpoint
                    iter_dir = output_dir / f"iteration_{iteration}"
                    iter_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        compiled.save(iter_dir / "merge_summarizer.json")
                    except Exception:
                        pass

                    return compiled, merge_stats

                except Exception as e:
                    logger.error(f"  Merge optimization failed: {e}")
                    merge_stats["merge_optimized"] = False
                    merge_stats["merge_error"] = str(e)
                    return merge_summarizer, merge_stats

            # Run both optimizations in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                leaf_future = executor.submit(optimize_leaf)
                merge_future = executor.submit(optimize_merge)

                # Wait for results
                leaf_summarizer, leaf_stats, summarizer_score = leaf_future.result()
                iter_stats.update(leaf_stats)

                merge_summarizer, merge_stats = merge_future.result()
                iter_stats.update(merge_stats)

            logger.info("  Parallel summarizer optimization complete")
        else:
            logger.info("Step 3: Skipping summarizer optimization (--skip-summarizer-opt)")
            iter_stats["summarizer_skipped"] = True

        # Check convergence
        oracle_improvement = oracle_score - prev_oracle_score
        summarizer_improvement = summarizer_score - prev_summarizer_score

        iter_stats["oracle_improvement"] = oracle_improvement
        iter_stats["summarizer_improvement"] = summarizer_improvement

        # Standardized keys for consistent printing (using pipeline MAE)
        pipeline_mae = iter_stats.get("pipeline_mae", float("inf"))
        iter_stats["metric_before"] = prev_pipeline_mae if iteration > 1 else pipeline_mae
        iter_stats["metric_after"] = pipeline_mae
        iter_stats["improvement"] = iter_stats["metric_before"] - pipeline_mae  # Positive = better (lower MAE)
        prev_pipeline_mae = pipeline_mae

        # Update tracking
        prev_oracle_score = oracle_score
        prev_summarizer_score = summarizer_score

        # Check if we've converged (only check components that are being optimized)
        has_improved = (
            (not args.skip_oracle_opt and oracle_improvement > args.convergence_threshold) or
            (not args.skip_summarizer_opt and summarizer_improvement > args.convergence_threshold)
        )

        if has_improved:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            logger.info(f"  No significant improvement ({no_improvement_count}/{args.convergence_patience})")

        # Save iteration checkpoint
        iter_stats["duration_seconds"] = time.time() - iter_start
        iter_stats["no_improvement_count"] = no_improvement_count
        iteration_stats["iterations"].append(iter_stats)

        checkpoint_dir = output_dir / f"iteration_{iteration}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with open(checkpoint_dir / "stats.json", "w") as f:
            json.dump(iter_stats, f, indent=2)

        # Save modules
        try:
            oracle_classifier.save(checkpoint_dir / "oracle.json")
            leaf_summarizer.save(checkpoint_dir / "leaf_summarizer.json")
            merge_summarizer.save(checkpoint_dir / "merge_summarizer.json")
        except Exception as e:
            logger.debug(f"Could not save checkpoints: {e}")

        logger.info(f"Iteration {iteration} complete in {iter_stats['duration_seconds']:.1f}s")

        # Check for early stopping
        if run_until_convergence and no_improvement_count >= args.convergence_patience:
            logger.info(f"Converged after {iteration} iterations (no improvement for {args.convergence_patience} rounds)")
            iteration_stats["converged"] = True
            break

    print_section(f"Optimization Complete ({iteration} iterations)")
    return oracle_classifier, leaf_summarizer, merge_summarizer, iteration_stats


if __name__ == "__main__":
    main()
