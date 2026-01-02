#!/usr/bin/env python3
"""
Document Training Pipeline for Oracle-Preserving Summarization.

Runs two-step iterative optimization: oracle + summarizer.

This script is called by scripts/run_training_pipeline.sh and provides
the main entry point for training the OPS summarization pipeline.

The pipeline is task-agnostic via the --task flag and dataset-agnostic via
the --dataset flag.

Example:
    python -m src.training.run_pipeline --port 8000 --train-samples 30
    python -m src.training.run_pipeline --task document_analysis --dataset jsonl --port 8000

"""

import os

# Set NumExpr thread limit before any imports that might load it
# This avoids the "detected N cores but limiting to M" warnings
os.environ.setdefault("NUMEXPR_MAX_THREADS", "64")

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dspy

from src.config.constants import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
from src.config.dspy_config import configure_dspy

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments matching the shell script interface."""
    parser = argparse.ArgumentParser(
        description='Document Training Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Server options
    parser.add_argument('--port', type=int, default=8000,
                        help='vLLM server port')
    parser.add_argument('--opt-model-port', type=int, default=None,
                        help='Separate port for optimization model (optional)')

    # Data options
    parser.add_argument('--train-samples', type=int, default=33,
                        help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=11,
                        help='Number of validation samples')
    parser.add_argument('--test-samples', type=int, default=11,
                        help='Number of test samples')

    # Concurrency options
    parser.add_argument('--concurrent-docs', type=int, default=20,
                        help='Documents to process in parallel')
    parser.add_argument('--concurrent-requests', type=int, default=200,
                        help='Concurrent LLM requests')
    parser.add_argument('--num-threads', type=int, default=64,
                        help='Parallel metric evaluations')

    # Optimizer options
    parser.add_argument('--optimizer', type=str, default='bootstrap_random_search',
                        choices=['gepa', 'bootstrap', 'bootstrap_random_search',
                                 'mipro', 'labeled_fewshot'],
                        help='Optimizer type')
    parser.add_argument('--optimizer-budget', type=str, default='heavy',
                        choices=['light', 'medium', 'heavy'],
                        help='Budget level for GEPA/MIPRO')
    parser.add_argument('--max-metric-calls', type=int, default=None,
                        help='Direct control over metric calls (overrides budget)')

    # Iterative optimization
    parser.add_argument('--n-iterations', type=int, default=1,
                        help='Number of iterations (1=single-pass, 2+=iterative, 0=until convergence)')
    parser.add_argument('--convergence-threshold', type=float, default=0.01,
                        help='Threshold for early stopping')
    parser.add_argument('--convergence-patience', type=int, default=3,
                        help='Rounds without improvement before stopping')
    parser.add_argument('--skip-oracle-opt', action='store_true',
                        help='Skip oracle/scorer optimization')

    # GenRM OPS Tree Building
    # Builds trees with tournament selection, collecting demos and preferences
    # NOTE: For initialization with demo seeding, use --enable-genrm (replaces old top-down-init)
    parser.add_argument('--enable-genrm', action='store_true',
                        help='Enable GenRM for OPS tree building and preference collection')
    parser.add_argument('--max-init-prompt-tokens', type=int, default=4000,
                        help='Max tokens for init prompts (doc + rubric + instructions)')
    parser.add_argument('--genrm-port', type=int, default=8001,
                        help='Port for GenRM server')
    parser.add_argument('--genrm-init-samples', type=int, default=8,
                        help='Number of OPS trees to build')
    parser.add_argument('--genrm-init-candidates', type=int, default=4,
                        help='Candidates per node for GenRM tournament')
    parser.add_argument('--train-comparison-module', action='store_true',
                        help='Train OPSComparisonModule from collected preferences')

    # Tournament of Tournaments (Judge Optimization)
    parser.add_argument('--optimize-judge', action='store_true',
                        help='Shorthand for --tournament-of-tournaments --tot-max-iterations 1 (single-pass judge optimization)')
    parser.add_argument('--judge-optimization-budget', type=str, default='light',
                        choices=['light', 'medium', 'heavy', 'superheavy'],
                        help='Budget for judge optimization (default: light)')
    parser.add_argument('--use-dspy-strategy', action='store_true',
                        help='Use DSPyStrategy for tree building (enables tournament + preference collection via strategy pattern)')
    parser.add_argument('--load-optimized-judge', type=str, default=None,
                        help='Path to load pre-optimized judge (skips judge optimization)')

    # Full Iterative Tournament of Tournaments Loop
    parser.add_argument('--tournament-of-tournaments', action='store_true',
                        help='Enable full iterative ToT loop (builds trees, optimizes judge, repeats until convergence)')
    parser.add_argument('--tot-max-iterations', type=int, default=5,
                        help='Maximum ToT iterations (default: 5)')
    parser.add_argument('--tot-convergence-threshold', type=float, default=0.01,
                        help='Stop if improvement below this (default: 0.01)')
    parser.add_argument('--tot-convergence-patience', type=int, default=2,
                        help='Stop after N iterations without improvement (default: 2)')
    parser.add_argument('--tot-samples-per-iteration', type=int, default=50,
                        help='Number of samples to process per ToT iteration (default: 50)')
    parser.add_argument('--tot-judge-test-split', type=float, default=0.2,
                        help='Holdout split for judge optimization (default: 0.2)')
    parser.add_argument('--tot-shuffle-samples', action=argparse.BooleanOptionalAction, default=True,
                        help='Shuffle samples each ToT iteration (default: True)')
    parser.add_argument('--tot-random-seed', type=int, default=42,
                        help='Random seed for ToT sample shuffling (default: 42)')

    # Resume and output
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint (skips completed phases)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')

    # Inference mode (skip training, use pre-trained scorer)
    parser.add_argument('--load-scorer-path', type=str, default=None,
                        help='Path to pre-trained scorer module (skips optimization)')
    parser.add_argument('--inference-only', action='store_true',
                        help='Run inference only (requires --load-scorer-path)')

    # Scale configuration (task-derived when available)
    parser.add_argument('--scale-min', type=float, default=None,
                        help='Minimum value of the scoring scale (required if task has no scale)')
    parser.add_argument('--scale-max', type=float, default=None,
                        help='Maximum value of the scoring scale (required if task has no scale)')

    # Task/dataset configuration
    parser.add_argument('--task', type=str, default=None,
                        help='Task plugin to use (default: settings.yaml tasks.default)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset plugin to use (default: settings.yaml datasets.default)')
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path for file-based datasets (e.g., jsonl)')

    return parser.parse_args()


def normalize_judge_optimization_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Unify --optimize-judge and --tournament-of-tournaments into a single path.

    When --optimize-judge is set without --tournament-of-tournaments, treat it
    as ToT with max_iterations=1. This avoids duplicate code paths and ensures
    consistent behavior.
    """
    if getattr(args, 'optimize_judge', False) and not getattr(args, 'tournament_of_tournaments', False):
        logger.info("--optimize-judge is shorthand for --tournament-of-tournaments --tot-max-iterations 1")
        args.tournament_of_tournaments = True
        # Only set max_iterations to 1 if user didn't explicitly set it
        if not hasattr(args, 'tot_max_iterations') or args.tot_max_iterations == 5:  # 5 is default
            args.tot_max_iterations = 1
    return args


def resolve_task_and_dataset(args: argparse.Namespace) -> Tuple[str, str, dict]:
    """Resolve task and dataset names from args and settings."""
    from src.config.settings import (
        load_settings,
        get_default_task,
        get_default_dataset,
        get_task_config,
        get_dataset_config,
    )

    settings = load_settings()
    task_name = args.task or get_default_task(settings)
    dataset_name = args.dataset or get_default_dataset(settings)

    task_config = get_task_config(task_name, settings)
    dataset_config = get_dataset_config(dataset_name, settings)
    return task_name, dataset_name, {"settings": settings, "task": task_config, "dataset": dataset_config}


def setup_dspy(args: argparse.Namespace) -> None:
    """Configure DSPy with the vLLM server."""
    model_url = f"http://localhost:{args.port}/v1"

    # Get model name from server
    try:
        import requests
        response = requests.get(f"{model_url}/models", timeout=5)
        model_info = response.json()
        model_name = model_info['data'][0]['id'] if model_info.get('data') else 'default'
    except Exception as e:
        logger.warning(f"Could not get model name from server: {e}")
        model_name = 'default'

    logger.info(f"Configuring DSPy with model: {model_name}")

    # Configure DSPy LM
    # Nemotron/GenRM models support 32768+ tokens, use 16384 for output to leave room for input
    lm = dspy.LM(
        model=f"openai/{model_name}",
        api_base=model_url,
        api_key="EMPTY",
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    configure_dspy(lm=lm)


def create_prompt_lm(args: argparse.Namespace) -> tuple[Optional[dspy.LM], Optional[str]]:
    """Create a separate LM for prompt optimization (optional)."""
    if args.opt_model_port is None:
        return None, None

    from src.config.settings import load_settings
    from src.core.model_detection import detect_model_from_port

    settings = load_settings()
    gen_cfg = settings.get('generation', {})
    prompt_cfg = gen_cfg.get('comparison_judge', {})

    model_name = detect_model_from_port(port=args.opt_model_port)
    lm = dspy.LM(
        model=f"openai/{model_name}",
        api_base=f"http://localhost:{args.opt_model_port}/v1",
        api_key="EMPTY",
        temperature=prompt_cfg.get('temperature', 0.3),
        max_tokens=prompt_cfg.get('max_tokens', 16384),
    )
    return lm, model_name


def save_prompt_context(
    judge: Any,
    output_dir: Path,
    rubric: str,
    law_types: List[str],
    source: str,
) -> Optional[Path]:
    """Persist prompt-tuned GenRM context for inspection."""
    if judge is None or not getattr(judge, "use_dspy_prompt", False):
        return None

    report = {
        "source": source,
        "rubric": rubric,
        "law_types": {},
        "created_at": datetime.now().isoformat(),
    }

    for law_type in law_types:
        try:
            extra_context = judge.get_prompt_context(rubric, law_type)
            report["law_types"][law_type] = {
                "extra_context": extra_context,
            }
        except Exception as e:
            report["law_types"][law_type] = {
                "extra_context": "",
                "error": str(e),
            }

    prompt_dir = output_dir / "optimized_judge"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / "prompt_context.json"
    with open(prompt_path, "w") as f:
        json.dump(report, f, indent=2)

    return prompt_path


def build_trees(
    train_results: List[Any],
    train_samples: List[Any],
    args: argparse.Namespace,
    task: Optional[Any] = None,
    output_dir: Path = None,
    judge_override: Optional[Any] = None,
) -> Tuple[List[Any], Any, List[dspy.Example]]:
    """
    Build OPS trees and collect preferences using GenRM.

    Filters to init documents whose full summarizer prompt fits within the
    init prompt budget (doc + rubric + instructions).

    Args:
        train_results: List of processed document results from Phase 1
        train_samples: Original document samples (for accessing original text)
        args: Command line arguments
        output_dir: Output directory for saving preferences
        judge_override: Optional judge to use for tournament selection

    Returns:
        Tuple of (trees, preferences, demos)
    """
    from datetime import datetime
    from src.tree.builder import TreeBuilder, BuildConfig
    from src.training.preference import GenRMJudge
    from src.training.preference import PreferenceDataset
    from src.core.strategy import CallableStrategy, TournamentStrategy, TournamentConfig
    from src.core.model_detection import detect_model_from_port
    from src.config.settings import load_settings
    from src.tasks import get_task

    # Load task plugin for rubric/context
    if task is None:
        task = get_task(args.task)
    logger.info(f"Using task: {task.name}")

    logger.info("Building OPS trees with GenRM validation...")

    # Load settings
    settings = load_settings()
    gen_cfg = settings.get('generation', {})
    summarizer_cfg = gen_cfg.get('summarizer', {})
    judge_cfg = gen_cfg.get('genrm_judge', {})

    # Configure summarizer LM (main model on args.port)
    summarizer_model_name = detect_model_from_port(port=args.port)
    logger.info(f"  Summarizer model: {summarizer_model_name}")

    summarizer_lm = dspy.LM(
        model=f"openai/{summarizer_model_name}",
        api_base=f"http://localhost:{args.port}/v1",
        api_key="EMPTY",
        temperature=summarizer_cfg.get('temperature', 0.5),
        max_tokens=summarizer_cfg.get('max_tokens', 8192),
    )
    configure_dspy(lm=summarizer_lm)

    # Create summarizer module for tree building
    summarizer = task.create_summarizer()

    # Create GenRM judge (auto-detects model from server)
    judge = judge_override
    if judge is None:
        logger.info(f"  GenRM judge on port {args.genrm_port}")
        judge = GenRMJudge(
            base_url=f"http://localhost:{args.genrm_port}/v1",
            model_name=None,  # Auto-detect
            temperature=judge_cfg.get('temperature', 0.6),
            top_p=judge_cfg.get('top_p', 0.95),
            max_tokens=judge_cfg.get('max_tokens', 8192),
        )
    else:
        logger.info("  Using provided judge override for tournament selection")

    # Get rubric from task plugin (task-agnostic)
    rubric = task.create_rubric()
    prompt_builders = task.create_prompt_builders()
    summarize_prompt_fn = prompt_builders.summarize
    k_candidates = args.genrm_init_candidates
    n_samples = args.genrm_init_samples
    init_prompt_token_limit = args.max_init_prompt_tokens
    from src.preprocessing.tokenizer import TokenCounter
    token_counter = TokenCounter(model=summarizer_model_name)

    def _count_prompt_tokens(text: str) -> int:
        messages = summarize_prompt_fn(text, rubric)
        prompt_text = "\n".join(
            f"{msg.get('role', '')}: {msg.get('content', '')}"
            for msg in messages
            if isinstance(msg, dict)
        )
        return token_counter.count(prompt_text)

    # Create lookup from doc_id to original sample text
    sample_text_lookup = {s.doc_id: s.text for s in train_samples}

    # Collect init segments whose full prompt fits in the context budget
    segments = []
    skipped_count = 0
    for result in train_results:
        if result is None or getattr(result, 'error', None) is not None:
            skipped_count += 1
            continue

        doc_id = getattr(result, 'doc_id', 'unknown')

        # Get original text from samples lookup (avoids storing text on result)
        text = sample_text_lookup.get(doc_id)
        if not text:
            logger.debug(f"Skipping result {doc_id}: no matching sample found")
            skipped_count += 1
            continue

        reference_score = getattr(result, 'reference_score', None)

        prompt_tokens = _count_prompt_tokens(text)
        if prompt_tokens <= init_prompt_token_limit:
            segments.append({
                'text': text,
                'doc_id': doc_id,
                'reference_score': reference_score,
            })
        else:
            logger.debug(
                f"Skipping {doc_id}: init prompt too long "
                f"({prompt_tokens} > {init_prompt_token_limit} tokens)"
            )
            skipped_count += 1

    if not segments:
        logger.warning(
            "No suitable init segments found for tree building "
            f"(skipped {skipped_count}/{len(train_results)}). "
            "Proceeding without GenRM init trees."
        )
        return [], PreferenceDataset(), []

    if skipped_count > 0:
        logger.info(f"Using {len(segments)} segments for tree building (skipped {skipped_count} unsuitable)")

    # Sample segments
    import random
    random.seed(42)
    samples = random.sample(segments, min(len(segments), n_samples))

    logger.info(f"  Init prompt budget: {init_prompt_token_limit} tokens (doc + rubric + instructions)")
    logger.info(f"  Building {len(samples)} init trees")
    logger.info(f"  K candidates: {k_candidates}")

    # Build trees using TreeBuilder + TournamentStrategy (consolidated path)
    config = BuildConfig(k=k_candidates)
    base_strategy = CallableStrategy(summarizer)
    tournament_strategy = TournamentStrategy(
        base=base_strategy,
        judge=judge,
        config=TournamentConfig(k=k_candidates),
    )
    builder = TreeBuilder(strategy=tournament_strategy, config=config)

    trees = []
    all_demos = []
    all_preferences = PreferenceDataset()

    for i, segment in enumerate(samples):
        try:
            # Build tree using new unified API
            result = builder.build_sync(segment['text'], rubric)
            tree = result.tree

            # Store metadata for tracking
            tree.metadata['doc_id'] = segment['doc_id']
            tree.metadata['reference_score'] = segment['reference_score']

            trees.append(tree)
            for pref in result.preferences:
                pref.source_example_id = segment['doc_id']
                pref.reference_score = segment['reference_score']
            all_preferences.add_pairs(result.preferences)

            # Create demos: pair leaves with final summary
            for leaf in tree.leaves:
                all_demos.append(dspy.Example(
                    content=leaf.raw_text_span,
                    rubric=rubric,
                    summary=tree.final_summary,
                ).with_inputs("content", "rubric"))

            logger.debug(f"  Tree {i+1}/{len(samples)}: {tree}")

            # Reset preferences for next tree
            builder.reset()

        except Exception as e:
            logger.warning(f"  Failed to build tree for {segment['doc_id']}: {e}")
            continue

        # Progress logging
        if (i + 1) % 5 == 0 or (i + 1) == len(samples):
            logger.info(f"  Progress: {i+1}/{len(samples)} trees, {len(all_preferences)} preferences")

    # Save preferences if output_dir provided
    if output_dir:
        prefs_dir = output_dir / 'preferences'
        prefs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pref_file = prefs_dir / f"preferences_ops_tree_{timestamp}.json"
        all_preferences.save(pref_file)

        # Save DPO format
        dpo_data = all_preferences.to_dpo_format()
        dpo_file = prefs_dir / f"dpo_ops_tree_{timestamp}.json"
        with open(dpo_file, 'w') as f:
            json.dump(dpo_data, f, indent=2)

        # Save tree stats
        tree_stats = {
            'n_trees': len(trees),
            'n_preferences': len(all_preferences),
            'n_demos': len(all_demos),
            'init_prompt_token_limit': init_prompt_token_limit,
            'k_candidates': k_candidates,
            'tree_summaries': [
                {
                    'doc_id': t.metadata.get('doc_id'),
                    'height': t.height,
                    'node_count': t.node_count,
                    'leaf_count': t.leaf_count,
                }
                for t in trees
            ],
        }
        stats_file = prefs_dir / f"tree_stats_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(tree_stats, f, indent=2)

        logger.info(f"  Saved preferences to: {pref_file}")

    logger.info(f"\nOPS Tree Building Complete:")
    logger.info(f"  Trees built: {len(trees)}")
    logger.info(f"  Preferences collected: {len(all_preferences)}")
    logger.info(f"  Demos extracted: {len(all_demos)}")

    return trees, all_preferences, all_demos


def load_doc_data(
    args: argparse.Namespace,
    dataset: Any,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Load document dataset and split into train/val/test."""
    logger.info("Loading document dataset...")

    all_samples = dataset.load_samples(
        path=args.dataset_path,
        limit=args.train_samples + args.val_samples + args.test_samples,
        shuffle=True,
        seed=42,
    )

    logger.info(f"Loaded {len(all_samples)} total samples from dataset '{dataset.name}'")

    train_end = args.train_samples
    val_end = train_end + args.val_samples
    test_end = val_end + args.test_samples

    train_samples = all_samples[:train_end]
    val_samples = all_samples[train_end:val_end]
    test_samples = all_samples[val_end:test_end]

    logger.info(f"Split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    return train_samples, val_samples, test_samples


def normalize_samples_scores(samples: List[Any], task: Any) -> None:
    """Normalize reference scores on samples in-place to 0-1."""
    for sample in samples:
        if sample is None:
            continue
        raw = getattr(sample, "reference_score", None)
        if raw is None:
            continue
        try:
            normalized = task.normalize_score(raw)
        except Exception:
            continue
        sample.reference_score = normalized
        metadata = getattr(sample, "metadata", None)
        if isinstance(metadata, dict):
            metadata["score_normalized"] = True


def normalize_result_scores(results: List[Any], task: Any) -> None:
    """Normalize result scores in-place to 0-1 when needed."""
    for result in results:
        if result is None:
            continue
        metadata = getattr(result, "metadata", None)
        already_normalized = isinstance(metadata, dict) and metadata.get("score_normalized")
        if already_normalized:
            continue

        for field in ("reference_score", "estimated_score", "baseline_score"):
            value = getattr(result, field, None)
            if value is None:
                continue
            if 0.0 <= float(value) <= 1.0:
                continue
            try:
                setattr(result, field, task.normalize_score(value))
            except Exception:
                continue

        if isinstance(metadata, dict):
            metadata["score_normalized"] = True


def process_docs(
    samples: List[Any],
    args: argparse.Namespace,
    task: Any,
    desc: str = "Processing"
) -> List[Any]:
    """Process document samples through the batched OPS pipeline.

    Uses BatchedDocPipeline for high-throughput parallel processing:
    - Multiple documents processed concurrently
    - Level-wise batching for optimal GPU utilization
    - All LLM requests pooled and batched
    """
    import time
    from src.pipelines.batched import BatchedDocPipeline, BatchedPipelineConfig

    logger.info(f"{desc} {len(samples)} documents (batched mode)...")
    logger.info(f"  Concurrent docs: {args.concurrent_docs}, Concurrent requests: {args.concurrent_requests}")
    start_time = time.time()

    # Create batched pipeline config
    prompt_builders = task.create_prompt_builders()
    pipeline_config = BatchedPipelineConfig(
        task_model_url=f"http://localhost:{args.port}/v1",
        max_concurrent_documents=args.concurrent_docs,
        max_concurrent_requests=args.concurrent_requests,
        show_progress=True,
        rubric=task.create_rubric(),
        task_context=task.get_task_context(),
        prompt_builders=prompt_builders,
        score_parser=task.parse_score,
    )

    # Create batched pipeline
    pipeline = BatchedDocPipeline(config=pipeline_config)

    # Process samples in batched mode (sync wrapper runs async internally)
    results = pipeline.process_batch(samples)

    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r is not None and getattr(r, 'error', None) is None)

    logger.info(f"{desc} complete: {successful}/{len(samples)} successful in {elapsed:.1f}s")
    if len(samples) > 0:
        logger.info(f"  Throughput: {len(samples) / elapsed:.2f} docs/sec")

    return results


def run_optimization(
    train_results: List[Any],
    val_results: List[Any],
    args: argparse.Namespace,
    output_dir: Path,
    task: Any,
    init_demos: Optional[List[dspy.Example]] = None,
) -> Tuple[Dict[str, Any], Any]:
    """Run the oracle/summarizer optimization loop.

    Args:
        train_results: Processed document results from Phase 1 (training set)
        val_results: Processed document results from Phase 1 (validation set)
        args: Command-line arguments
        output_dir: Output directory
        init_demos: Optional list of dspy.Example demos to seed modules (from GenRM)

    Returns:
        Tuple of (optimization statistics dict, trained scorer module)
    """
    from src.training.core import OptimizationConfig
    from src.training.optimization import get_optimizer
    from src.core.scoring import UNIT_SCALE

    logger.info("Starting optimization...")

    logger.info(f"Using task: {task.name}")

    # If a separate optimization model port is specified, configure DSPy to use it
    if args.opt_model_port is not None:
        opt_port = args.opt_model_port
        logger.info(f"Using separate optimization model on port {opt_port}")
        try:
            import requests
            opt_model_url = f"http://localhost:{opt_port}/v1"
            response = requests.get(f"{opt_model_url}/models", timeout=5)
            opt_model_info = response.json()
            opt_model_name = opt_model_info['data'][0]['id'] if opt_model_info.get('data') else 'default'
            logger.info(f"Optimization model: {opt_model_name}")

            opt_lm = dspy.LM(
                model=f"openai/{opt_model_name}",
                api_base=opt_model_url,
                api_key="EMPTY",
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            dspy.configure(lm=opt_lm)
        except Exception as e:
            logger.warning(f"Could not configure optimization model: {e}, using default")

    # Internal optimization uses normalized 0-1 scale
    scale = UNIT_SCALE
    if task.scale is not None:
        logger.info(f"Using normalized scale for task '{task.scale.name}': [0.0, 1.0]")
    else:
        logger.info("Using normalized scale: [0.0, 1.0]")

    # Create optimization config
    opt_config = OptimizationConfig(
        optimizer_type=args.optimizer,
        gepa_auto=args.optimizer_budget,
        mipro_auto=args.optimizer_budget,
        max_metric_calls=args.max_metric_calls,
        num_threads=args.num_threads,
        checkpoint_dir=output_dir / 'checkpoints',
    )

    # Create trainsets
    train_examples = task.create_trainset(train_results)
    val_examples = task.create_trainset(val_results)

    logger.info(f"Created {len(train_examples)} train, {len(val_examples)} val examples")

    # Initialize scorer using task plugin
    # The task's create_predictor returns the appropriate scorer for that task
    scorer = task.create_predictor()
    logger.info(f"Created scorer using task '{task.name}': {type(scorer).__name__}")

    if len(train_examples) < 4:
        logger.warning("Not enough training examples for optimization")
        logger.warning("Returning untrained scorer for test evaluation")
        return {'error': 'insufficient_training_data', 'scorer_trained': False}, scorer

    # Seed scorer with demos if available
    if init_demos and len(init_demos) > 0:
        logger.info(f"Seeding scorer with {len(init_demos)} demos")
        if hasattr(scorer, 'demos'):
            scorer.demos = init_demos
            logger.info(f"  Seeded scorer.demos with {len(init_demos)} demos")
        else:
            logger.warning(f"  Could not seed scorer - no demos attribute")

    # Create score prediction metric (task-agnostic)
    # This metric compares predicted score to reference_score
    # Use task's output_field_name for prediction access
    score_field = task.output_field_name

    def score_prediction_metric(example, prediction, trace=None, pred_name=None, pred_trace=None) -> float:
        """
        Score prediction metric.

        Compares predicted score to reference_score on a 0-1 scale.
        Score = 1 - |predicted - reference|
        """
        try:
            # Get reference score from example
            reference = getattr(example, 'reference_score', None)
            if reference is None:
                return 0.0
            reference = float(reference)

            # Get predicted score from prediction using task's output_field_name
            if isinstance(prediction, dict):
                predicted = prediction.get(score_field, prediction.get('score', 0.0))
            else:
                predicted = getattr(prediction, score_field, None)
                if predicted is None:
                    predicted = getattr(prediction, 'score', None)
                if predicted is None:
                    # Try accessing as dict-like
                    try:
                        predicted = prediction[score_field]
                    except (KeyError, TypeError):
                        try:
                            predicted = prediction['score']
                        except (KeyError, TypeError):
                            return 0.0

            predicted = float(predicted)

            # Compute normalized error score using scale
            # Score = 1 - |error| / scale.range
            score = scale.values_to_score(predicted, reference)

            return score

        except Exception as e:
            logger.debug(f"Metric evaluation error: {e}")
            return 0.0

    # Run optimization
    stats = {'rounds': []}

    n_iterations = args.n_iterations
    if n_iterations == 0:
        n_iterations = 100  # Cap for "until convergence"

    best_metric = float('-inf')  # Start low for higher-is-better metrics
    patience_counter = 0

    for iteration in range(n_iterations):
        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {iteration + 1}")
        logger.info(f"{'='*60}")

        round_stats = {'round': iteration + 1}

        # Optimize scorer using registry optimizer system
        if not args.skip_oracle_opt:
            logger.info(f"Optimizing score predictor using '{args.optimizer}' optimizer...")
            try:
                # Get optimizer from registry (uses args.optimizer type)
                optimizer = get_optimizer(args.optimizer, opt_config)

                # Evaluate metric before optimization
                metric_before = 0.0
                metric_before_count = 0
                for ex in val_examples[:min(10, len(val_examples))]:
                    try:
                        pred = scorer(text=ex.summary, task_context=ex.rubric)
                        metric_before += score_prediction_metric(ex, pred)
                        metric_before_count += 1
                    except Exception as e:
                        logger.warning(f"Metric eval (before) failed for example: {e}")
                if metric_before_count == 0:
                    logger.error("All metric evaluations failed before optimization")
                metric_before = metric_before / max(1, metric_before_count)

                # Run optimization using registry optimizer's compile() method
                scorer = optimizer.compile(
                    student=scorer,
                    trainset=train_examples,
                    valset=val_examples,
                    metric=score_prediction_metric,
                )

                # Evaluate metric after optimization
                metric_after = 0.0
                metric_after_count = 0
                for ex in val_examples[:min(10, len(val_examples))]:
                    try:
                        pred = scorer(text=ex.summary, task_context=ex.rubric)
                        metric_after += score_prediction_metric(ex, pred)
                        metric_after_count += 1
                    except Exception as e:
                        logger.warning(f"Metric eval (after) failed for example: {e}")
                if metric_after_count == 0:
                    logger.error("All metric evaluations failed after optimization")
                metric_after = metric_after / max(1, metric_after_count)

                round_stats['metric_before'] = metric_before
                round_stats['metric_after'] = metric_after
                round_stats['optimizer_used'] = args.optimizer
                logger.info(f"Scorer optimization: {metric_before:.4f} -> {metric_after:.4f}")
            except Exception as e:
                logger.error(f"Scorer optimization failed: {e}")
                raise

        stats['rounds'].append(round_stats)

        # Check convergence (higher metric is better)
        current_metric = round_stats.get('metric_after', best_metric)
        improvement = current_metric - best_metric  # Positive when improving

        if improvement < args.convergence_threshold:
            patience_counter += 1
            logger.info(f"No significant improvement (patience: {patience_counter}/{args.convergence_patience})")
            if patience_counter >= args.convergence_patience:
                logger.info("Convergence reached")
                break
        else:
            best_metric = current_metric
            patience_counter = 0

        # Save checkpoint
        checkpoint_path = output_dir / 'checkpoints' / f'iteration_{iteration + 1}.json'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, 'w') as f:
            json.dump(round_stats, f, indent=2)

    return stats, scorer


def write_score_report(
    rows: List[Dict[str, Any]],
    output_dir: Optional[Path],
    split_name: str,
) -> Optional[Path]:
    """Write per-document score report (raw + normalized) as JSONL."""
    if not output_dir:
        return None
    report_path = output_dir / f"{split_name}_score_report.jsonl"
    with open(report_path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return report_path


def evaluate_on_test(
    test_results: List[Any],
    scorer: Any,
    args: argparse.Namespace,
    task: Any,
    output_dir: Optional[Path] = None,
    split_name: str = "test",
) -> Dict[str, Any]:
    """Evaluate on a split with normalized (0-1) metrics."""
    logger.info(f"Evaluating on {split_name} set...")

    test_examples = task.create_trainset(test_results)

    if not test_examples:
        return {'error': 'no_test_examples'}

    # Collect predictions with error tracking
    results_with_errors = []
    failures = 0

    for idx, ex in enumerate(test_examples):
        try:
            result = scorer(text=ex.summary, task_context=ex.rubric)
            pred_score = float(result.get(task.output_field_name, result.get('score', 0)))
            true_score = float(ex.reference_score)
            error = abs(pred_score - true_score)
            doc_id = getattr(ex, 'original_content', None) or getattr(ex, 'doc_id', None) or f"example_{idx}"
            results_with_errors.append({
                'doc_id': doc_id,
                'example': ex,
                'predicted': pred_score,
                'actual': true_score,
                'error': error,
            })
        except Exception as e:
            logger.warning(f"Prediction failed for example: {e}")
            failures += 1

    if not results_with_errors:
        return {'error': 'no_valid_predictions', 'failures': failures}

    errors = [r['error'] for r in results_with_errors]

    # Normalized thresholds (0-1 scale)
    threshold_5pct = 0.05
    threshold_10pct = 0.10

    # Compute comprehensive metrics (normalized scale)
    mae = sum(errors) / len(errors)
    metrics = {
        'mae': mae,
        'mae_normalized': mae,
        'within_5pct': sum(1 for e in errors if e <= threshold_5pct) / len(errors) * 100,
        'within_10pct': sum(1 for e in errors if e <= threshold_10pct) / len(errors) * 100,
        'max_error': max(errors),
        'min_error': min(errors),
        'n_examples': len(test_examples),
        'n_evaluated': len(results_with_errors),
        'n_failures': failures,
    }
    metrics['within_5pct_normalized'] = metrics['within_5pct']
    metrics['within_10pct_normalized'] = metrics['within_10pct']
    metrics['max_error_normalized'] = metrics['max_error']
    metrics['min_error_normalized'] = metrics['min_error']

    report_rows = []
    for r in results_with_errors:
        report_rows.append({
            'doc_id': r['doc_id'],
            'predicted': r['predicted'],
            'actual': r['actual'],
            'error': r['error'],
        })
    report_path = write_score_report(report_rows, output_dir, split_name)
    if report_path:
        metrics['report_path'] = str(report_path)

    # Log worst predictions for debugging
    sorted_results = sorted(results_with_errors, key=lambda x: x['error'], reverse=True)
    logger.info("Worst 5 predictions:")
    for r in sorted_results[:5]:
        logger.info(
            "  Pred=%.3f, Actual=%.3f, Error=%.3f",
            r['predicted'],
            r['actual'],
            r['error'],
        )

    return metrics


def save_results(
    stats: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Save final results to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save final stats
    stats_path = output_dir / 'final_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Results saved to {stats_path}")


def train_comparison_module(
    preference_dataset: Any,
    args: argparse.Namespace,
    output_dir: Path,
) -> Any:
    """
    Train OPSComparisonModule from collected preferences.

    Note: This function always uses GEPA optimization regardless of the
    --optimizer flag. GEPA is specifically suited for training comparison
    modules from preference data. The --optimizer flag controls the main
    score predictor optimization, not comparison module training.

    Args:
        preference_dataset: PreferenceDataset with collected pairs
        args: Command-line arguments
        output_dir: Output directory

    Returns:
        Trained OPSComparisonModule
    """
    from datetime import datetime
    from src.training.comparison import OPSComparisonModule

    logger.info("\n" + "-" * 60)
    logger.info("Training OPSComparisonModule from preferences")
    logger.info("-" * 60)

    # Define preference metric
    def preference_metric(example, prediction, trace=None) -> float:
        predicted = str(getattr(prediction, "preferred", "")).upper().strip()
        actual = str(getattr(example, "preferred", "")).upper().strip()
        if predicted == actual:
            return 1.0
        if predicted == "TIE" or actual == "TIE":
            return 0.5
        return 0.0

    # Split dataset
    train_set, val_set = preference_dataset.split(train_ratio=0.8, shuffle=True)
    train_examples = train_set.to_dspy_examples()
    val_examples = val_set.to_dspy_examples()

    logger.info(f"  Train examples: {len(train_examples)}")
    logger.info(f"  Val examples: {len(val_examples)}")

    if len(train_examples) < 10:
        logger.warning(f"  Only {len(train_examples)} examples, skipping comparison module training")
        return None

    # Create comparison module
    comparison_module = OPSComparisonModule(use_cot=True)

    # Create optimizer
    optimizer = dspy.GEPA(
        metric=preference_metric,
        auto=args.optimizer_budget,
        num_threads=args.num_threads,
    )

    # Compile (always uses GEPA for comparison module, regardless of --optimizer flag)
    logger.info(f"  Starting GEPA optimization for comparison module (budget: {args.optimizer_budget})...")
    logger.info(f"  Note: Comparison module training uses GEPA, main optimizer is '{args.optimizer}'")
    compile_kwargs = {
        "student": comparison_module,
        "trainset": train_examples,
    }
    if val_examples:
        compile_kwargs["valset"] = val_examples

    try:
        trained_module = optimizer.compile(**compile_kwargs)
    except Exception as e:
        logger.error(f"  Optimization failed: {e}")
        return None

    # Save trained module
    module_dir = output_dir / 'comparison_module'
    module_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = module_dir / f"ops_comparison_{timestamp}.json"
    trained_module.save(str(model_path))

    # Save stats
    stats = {
        "created_at": datetime.now().isoformat(),
        "model_path": str(model_path),
        "num_pairs": len(preference_dataset),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "config": {
            "optimizer": "gepa",  # Always GEPA for comparison module training
            "main_optimizer": args.optimizer,  # Main pipeline optimizer for reference
            "budget": args.optimizer_budget,
            "num_threads": args.num_threads,
        },
    }
    stats_path = module_dir / f"ops_comparison_{timestamp}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"  Saved trained module to {model_path}")

    return trained_module


def run_training_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Main training pipeline entry point.

    Args:
        args: Parsed command-line arguments

    Returns:
        Dictionary with training statistics
    """
    import pickle

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting training pipeline")
    logger.info(f"Output directory: {output_dir}")
    if args.resume:
        logger.info("Resume mode enabled - will skip completed phases")

    # Resolve task and dataset
    task_name, dataset_name, configs = resolve_task_and_dataset(args)
    from src.tasks import get_task
    from src.datasets import get_dataset

    task = get_task(task_name, **configs["task"])
    dataset = get_dataset(dataset_name, **configs["dataset"])

    logger.info(f"Using task: {task.name}")
    logger.info(f"Using dataset: {dataset.name}")

    # Persist resolved args for reproducibility
    args.task = task.name
    args.dataset = dataset.name

    # Save config
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Setup DSPy
    setup_dspy(args)

    # Load data
    train_samples, val_samples, test_samples = load_doc_data(args, dataset)
    normalize_samples_scores(train_samples, task)
    normalize_samples_scores(val_samples, task)
    normalize_samples_scores(test_samples, task)

    stats = {
        'started_at': datetime.now().isoformat(),
        'config': vars(args),
        'task': task.name,
        'dataset': dataset.name,
    }

    # Handle inference-only mode
    if args.inference_only:
        if not args.load_scorer_path:
            raise ValueError("--inference-only requires --load-scorer-path")
        logger.info("Inference-only mode: skipping training phases")

    # Check for existing checkpoints if resuming
    phase1_checkpoint = checkpoint_dir / 'phase1_complete.json'
    phase1_data_checkpoint = checkpoint_dir / 'phase1_data.pkl'
    phase1_5_checkpoint = checkpoint_dir / 'phase1_5_complete.json'
    phase1_5_data_checkpoint = checkpoint_dir / 'phase1_5_data.pkl'

    train_results = None
    val_results = None
    init_demos = None
    preference_dataset = None
    ops_trees = None

    try:
        # Phase 1: Process documents
        if args.resume and phase1_checkpoint.exists() and phase1_data_checkpoint.exists():
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 1: Loading from checkpoint (skipping processing)")
            logger.info("=" * 60)
            with open(phase1_data_checkpoint, 'rb') as f:
                phase1_data = pickle.load(f)
                train_results = phase1_data['train_results']
                val_results = phase1_data['val_results']
            normalize_result_scores(train_results, task)
            normalize_result_scores(val_results, task)
            logger.info(f"Loaded {len(train_results)} train, {len(val_results)} val results from checkpoint")
        else:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 1: Processing Documents")
            logger.info("=" * 60)

            train_results = process_docs(train_samples, args, task, "Train")
            val_results = process_docs(val_samples, args, task, "Val")
            normalize_result_scores(train_results, task)
            normalize_result_scores(val_results, task)

            # Save phase 1 checkpoint
            with open(phase1_checkpoint, 'w') as f:
                json.dump({
                    'train_count': len(train_results),
                    'val_count': len(val_results),
                }, f, indent=2)
            # Save data for resume
            with open(phase1_data_checkpoint, 'wb') as f:
                pickle.dump({
                    'train_results': train_results,
                    'val_results': val_results,
                }, f)

        # Phase 1.5: OPS Tree Building with GenRM (unified)
        # Builds trees with tournament selection, collecting both demos and preferences
        comparison_module = None

        if args.enable_genrm:
            # Check for resume from Phase 1.5
            if args.resume and phase1_5_checkpoint.exists() and phase1_5_data_checkpoint.exists():
                logger.info("\n" + "=" * 60)
                logger.info("PHASE 1.5: Loading from checkpoint (skipping tree building)")
                logger.info("=" * 60)
                with open(phase1_5_data_checkpoint, 'rb') as f:
                    phase1_5_data = pickle.load(f)
                    ops_trees = phase1_5_data.get('ops_trees', [])
                    preference_dataset = phase1_5_data.get('preference_dataset')
                    init_demos = phase1_5_data.get('init_demos', [])
                logger.info(f"Loaded {len(ops_trees)} trees, {len(init_demos)} demos from checkpoint")
            else:
                logger.info("\n" + "=" * 60)
                logger.info("PHASE 1.5: OPS Tree Building with GenRM")
                logger.info("=" * 60)

                # Build trees - this collects both demos and preferences in one pass
                ops_trees, preference_dataset, init_demos = build_trees(
                    train_results, train_samples, args, task, output_dir,
                )

                # Save checkpoint data for resume
                with open(phase1_5_data_checkpoint, 'wb') as f:
                    pickle.dump({
                        'ops_trees': ops_trees,
                        'preference_dataset': preference_dataset,
                        'init_demos': init_demos,
                    }, f)

            # Record stats
            stats['genrm_trees'] = {
                'n_trees': len(ops_trees) if ops_trees else 0,
                'n_preferences': len(preference_dataset) if preference_dataset else 0,
                'n_demos': len(init_demos) if init_demos else 0,
                'n_samples': args.genrm_init_samples,
                'n_candidates': args.genrm_init_candidates,
                'init_prompt_token_limit': args.max_init_prompt_tokens,
            }

            if preference_dataset:
                stats['preference_collection'] = preference_dataset.summary()

            # Save checkpoint metadata
            with open(phase1_5_checkpoint, 'w') as f:
                json.dump({
                    'n_trees': len(ops_trees) if ops_trees else 0,
                    'n_demos': len(init_demos) if init_demos else 0,
                    'n_preferences': len(preference_dataset) if preference_dataset else 0,
                    'tree_summaries': [
                        {'doc_id': t.metadata.get('doc_id'), 'height': t.height, 'nodes': t.node_count}
                        for t in ops_trees
                    ] if ops_trees else [],
                }, f, indent=2)

            # Optionally train comparison module from preferences
            if getattr(args, 'train_comparison_module', False) and preference_dataset and len(preference_dataset) > 20:
                comparison_module = train_comparison_module(
                    preference_dataset, args, output_dir
                )
                if comparison_module is not None:
                    stats['comparison_module_trained'] = True

        # Phase 1.6: Tournament of Tournaments (Full Iterative Loop)
        # Iteratively improves the judge by: build trees → enrich with oracle → optimize judge → repeat
        optimized_judge = None
        tot_result = None

        if getattr(args, 'tournament_of_tournaments', False):
            from src.training.tournament_loop import (
                TournamentOfTournamentsTrainer,
                ToTConfig,
                load_optimized_judge as load_tot_judge,
            )
            from src.training.preference import GenRMJudge
            from src.config.settings import load_settings
            from src.core.model_detection import detect_model_from_port

            logger.info("\n" + "=" * 60)
            logger.info("PHASE 1.6: Tournament of Tournaments (Full Iterative Loop)")
            logger.info("=" * 60)

            # Create summarizer for tree building
            settings = load_settings()
            gen_cfg = settings.get('generation', {})
            summarizer_cfg = gen_cfg.get('summarizer', {})
            judge_cfg = gen_cfg.get('genrm_judge', {})

            summarizer_model_name = detect_model_from_port(port=args.port)
            summarizer_lm = dspy.LM(
                model=f"openai/{summarizer_model_name}",
                api_base=f"http://localhost:{args.port}/v1",
                api_key="EMPTY",
                temperature=summarizer_cfg.get('temperature', 0.5),
                max_tokens=summarizer_cfg.get('max_tokens', 8192),
            )
            configure_dspy(lm=summarizer_lm)

            prompt_lm, prompt_model_name = create_prompt_lm(args)
            if prompt_lm is not None:
                logger.info(f"  Prompt optimization model: {prompt_model_name} (port {args.opt_model_port})")

            # Create summarizer function (wraps DSPy module for sync API)
            summarizer_module = task.create_summarizer()
            rubric = task.create_rubric()

            def summarizer_fn(content: str, rubric: str) -> str:
                """Sync summarizer function for ToT."""
                result = summarizer_module(content=content, rubric=rubric)
                return getattr(result, 'summary', str(result))

            # Create initial judge
            initial_judge = GenRMJudge(
                base_url=f"http://localhost:{args.genrm_port}/v1",
                model_name=None,
                temperature=judge_cfg.get('temperature', 0.6),
                top_p=judge_cfg.get('top_p', 0.95),
                max_tokens=judge_cfg.get('max_tokens', 8192),
            )

            # Create oracle scorer from task
            oracle_predict = task.create_oracle_scorer()

            # Create sample lookup for ToT
            sample_lookup = {s.doc_id: s for s in train_samples}
            tot_samples = []
            for result in train_results:
                if result is None or getattr(result, 'error', None):
                    continue
                doc_id = getattr(result, 'doc_id', 'unknown')
                sample = sample_lookup.get(doc_id)
                if sample and hasattr(sample, 'text'):
                    tot_samples.append({
                        'text': sample.text,
                        'doc_id': doc_id,
                        'reference_score': getattr(result, 'reference_score', None),
                    })

            if len(tot_samples) < 10:
                logger.warning(f"Only {len(tot_samples)} samples for ToT, may be insufficient")

            # Configure ToT with normalized 0-1 scale
            scale_range = 1.0
            normalized_tie_margin = 0.05

            preference_labeler = None
            if hasattr(task, 'create_preference_labeler'):
                try:
                    preference_labeler = task.create_preference_labeler()
                except Exception as e:
                    logger.warning(f"Preference labeler creation failed: {e}")

            tot_config = ToTConfig(
                max_iterations=getattr(args, 'tot_max_iterations', 5),
                min_iterations=1,
                convergence_threshold=getattr(args, 'tot_convergence_threshold', 0.01),
                convergence_patience=getattr(args, 'tot_convergence_patience', 2),
                k_candidates=args.genrm_init_candidates,
                n_samples_per_iteration=getattr(args, 'tot_samples_per_iteration', 50),
                candidate_temperature=0.9,
                judge_budget=getattr(args, 'judge_optimization_budget', 'medium'),
                num_threads=args.num_threads,
                judge_test_split=getattr(args, 'tot_judge_test_split', 0.2),
                tie_margin=normalized_tie_margin,
                normalize_errors=True,
                scale_range=scale_range,
                preference_labeler=preference_labeler,
                shuffle_samples_each_iteration=getattr(args, 'tot_shuffle_samples', True),
                random_seed=getattr(args, 'tot_random_seed', 42),
            )

            logger.info(f"  Max iterations: {tot_config.max_iterations}")
            logger.info(f"  Convergence threshold: {tot_config.convergence_threshold}")
            logger.info(f"  Samples per iteration: {tot_config.n_samples_per_iteration}")
            logger.info(f"  Judge budget: {tot_config.judge_budget}")
            logger.info(f"  Judge test split: {tot_config.judge_test_split:.2f}")
            logger.info(f"  Tie margin (normalized): {tot_config.tie_margin:.4f}")

            # Run Tournament of Tournaments
            trainer = TournamentOfTournamentsTrainer(
                summarizer=summarizer_fn,
                oracle_predict=oracle_predict,
                initial_judge=initial_judge,
                config=tot_config,
                output_dir=output_dir,
                prompt_lm=prompt_lm,
            )

            tot_result = trainer.train(tot_samples, rubric)

            # Record stats
            stats['tournament_of_tournaments'] = {
                'converged': tot_result.converged,
                'convergence_reason': tot_result.convergence_reason,
                'final_iteration': tot_result.final_iteration,
                'final_judge_accuracy': tot_result.final_judge_accuracy,
                'improvement_history': tot_result.improvement_history,
                'optimized_judge_path': str(tot_result.optimized_judge_path) if tot_result.optimized_judge_path else None,
            }

            if tot_result.optimized_judge_path:
                optimized_judge = load_tot_judge(
                    tot_result.optimized_judge_path,
                    base_url=f"http://localhost:{args.genrm_port}/v1",
                    prompt_lm=prompt_lm,
                )
                logger.info(f"\nToT Complete:")
                logger.info(f"  Converged: {tot_result.converged} ({tot_result.convergence_reason})")
                logger.info(f"  Final accuracy: {tot_result.final_judge_accuracy:.3f}")
                logger.info(f"  Iterations: {tot_result.final_iteration}")
                logger.info(f"  Judge saved to: {tot_result.optimized_judge_path}")

            if optimized_judge is not None:
                prompt_path = save_prompt_context(
                    optimized_judge,
                    output_dir,
                    rubric,
                    ["sufficiency", "merge", "idempotence"],
                    source="tournament_of_tournaments",
                )
                if prompt_path is not None:
                    logger.info(f"  Prompt context saved to: {prompt_path}")

                logger.info("\n" + "=" * 60)
                logger.info("PHASE 1.65: Rebuilding OPS Trees with Optimized Judge")
                logger.info("=" * 60)

                opt_ops_trees, opt_preference_dataset, opt_init_demos = build_trees(
                    train_results, train_samples, args, task, output_dir,
                    judge_override=optimized_judge,
                )

                stats['genrm_trees_optimized_judge'] = {
                    'n_trees': len(opt_ops_trees) if opt_ops_trees else 0,
                    'n_preferences': len(opt_preference_dataset) if opt_preference_dataset else 0,
                    'n_demos': len(opt_init_demos) if opt_init_demos else 0,
                    'n_samples': args.genrm_init_samples,
                    'n_candidates': args.genrm_init_candidates,
                    'init_prompt_token_limit': args.max_init_prompt_tokens,
                }

                if opt_preference_dataset:
                    stats['preference_collection_optimized_judge'] = opt_preference_dataset.summary()

                if opt_init_demos:
                    init_demos = opt_init_demos

        # Phase 1.75: Judge Optimization (Legacy - now handled by ToT)
        # NOTE: As of the unification, --optimize-judge sets tournament_of_tournaments=True,
        # so Phase 1.6 (ToT) handles judge optimization. This block is kept for backwards
        # compatibility but should rarely execute.
        elif getattr(args, 'optimize_judge', False) and preference_dataset and len(preference_dataset) > 20:
            from src.training.judge_optimization import JudgeOptimizer, JudgeOptimizationConfig

            logger.info("\n" + "=" * 60)
            logger.info("PHASE 1.75: Judge Optimization (Tournament of Tournaments)")
            logger.info("=" * 60)

            if getattr(args, 'load_optimized_judge', None):
                # Load pre-optimized judge
                logger.info(f"Loading pre-optimized judge from {args.load_optimized_judge}")
                from src.training.judge_optimization import load_optimized_judge
                prompt_lm, prompt_model_name = create_prompt_lm(args)
                if prompt_lm is not None:
                    logger.info(f"  Prompt optimization model: {prompt_model_name} (port {args.opt_model_port})")
                optimized_judge = load_optimized_judge(
                    Path(args.load_optimized_judge),
                    use_dspy_prompt=True,
                    prompt_lm=prompt_lm,
                )
                stats['judge_loaded_from'] = str(args.load_optimized_judge)
                logger.info("  Note: Loaded judge available for use in future phases")
                rubric = task.create_rubric()
                prompt_path = save_prompt_context(
                    optimized_judge,
                    output_dir,
                    rubric,
                    ["sufficiency", "merge", "idempotence"],
                    source="judge_optimization_loaded",
                )
                if prompt_path is not None:
                    logger.info(f"  Prompt context saved to: {prompt_path}")
                # TODO: Wire optimized_judge into tree rebuilding for full tournament-of-tournaments loop
            else:
                # Optimize judge from preferences
                judge_config = JudgeOptimizationConfig(
                    budget=getattr(args, 'judge_optimization_budget', 'light'),
                    num_threads=args.num_threads,
                    checkpoint_dir=checkpoint_dir,
                )

                judge_optimizer = JudgeOptimizer(config=judge_config)

                # Convert PreferenceDataset to list if needed
                pref_list = list(preference_dataset) if hasattr(preference_dataset, '__iter__') else []
                from src.training.preference.genrm_dspy import GenRMComparisonModule
                prompt_lm, prompt_model_name = create_prompt_lm(args)
                if prompt_lm is not None:
                    logger.info(f"  Prompt optimization model: {prompt_model_name} (port {args.opt_model_port})")

                prompt_tuned_judge = GenRMComparisonModule(
                    use_dspy_prompt=True,
                    prompt_lm=prompt_lm,
                )
                if prompt_lm is not None:
                    with dspy.context(lm=prompt_lm):
                        optimized_judge, judge_results = judge_optimizer.optimize(
                            pref_list,
                            initial_judge=prompt_tuned_judge,
                        )
                else:
                    optimized_judge, judge_results = judge_optimizer.optimize(
                        pref_list,
                        initial_judge=prompt_tuned_judge,
                    )

                # Save optimized judge
                judge_path = output_dir / 'optimized_judge' / 'judge.json'
                judge_path.parent.mkdir(parents=True, exist_ok=True)
                judge_optimizer.save(optimized_judge, judge_path)

                stats['judge_optimization'] = judge_results
                stats['optimized_judge_path'] = str(judge_path)
                logger.info(f"Judge optimization complete. Improvement: {judge_results.get('improvement', 0):+.3f}")
                logger.info(f"  Optimized judge saved to: {judge_path}")
                logger.info(f"  To use in subsequent runs: --load_optimized_judge {judge_path}")

                rubric = task.create_rubric()
                prompt_path = save_prompt_context(
                    optimized_judge,
                    output_dir,
                    rubric,
                    ["sufficiency", "merge", "idempotence"],
                    source="judge_optimization",
                )
                if prompt_path is not None:
                    logger.info(f"  Prompt context saved to: {prompt_path}")

        # Phase 2: Optimization (or load pre-trained scorer)
        logger.info("\n" + "=" * 60)

        if args.load_scorer_path:
            # Load pre-trained scorer, skip optimization
            logger.info("PHASE 2: Loading Pre-trained Scorer")
            logger.info("=" * 60)
            logger.info(f"Loading scorer from {args.load_scorer_path}")

            trained_scorer = task.create_scorer()
            try:
                trained_scorer.load(str(args.load_scorer_path))
                stats['scorer_loaded_from'] = str(args.load_scorer_path)
                logger.info("Successfully loaded pre-trained scorer")
            except Exception as e:
                logger.error(f"Failed to load scorer: {e}")
                raise
        else:
            # Run optimization
            logger.info("PHASE 2: Optimization")
            logger.info("=" * 60)

            opt_stats, trained_scorer = run_optimization(
                train_results, val_results, args, output_dir, task,
                init_demos=init_demos
            )
            stats.update(opt_stats)

            # Save trained scorer
            if trained_scorer is not None:
                scorer_dir = output_dir / 'trained_modules'
                scorer_dir.mkdir(parents=True, exist_ok=True)
                scorer_path = scorer_dir / 'scorer_final.json'
                try:
                    trained_scorer.save(str(scorer_path))
                    stats['scorer_module_path'] = str(scorer_path)
                    logger.info(f"Saved trained scorer to {scorer_path}")
                except Exception as e:
                    logger.warning(f"Failed to save trained scorer: {e}")

        # Phase 3: Train/Test evaluation
        if trained_scorer is not None:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 3: Train/Test Evaluation")
            logger.info("=" * 60)

            train_eval = evaluate_on_test(
                train_results,
                trained_scorer,
                args,
                task,
                output_dir=output_dir,
                split_name="train",
            )
            stats['train'] = train_eval
            if 'error' not in train_eval:
                logger.info("Train Results:")
                logger.info(
                    "  MAE: %.3f",
                    train_eval.get('mae', 0),
                )
                logger.info(
                    "  Within 5%%: %.1f%%",
                    train_eval.get('within_5pct', 0),
                )
                logger.info(
                    "  Within 10%%: %.1f%%",
                    train_eval.get('within_10pct', 0),
                )
                logger.info(
                    "  Evaluated: %s/%s",
                    train_eval.get('n_evaluated', 0),
                    train_eval.get('n_examples', 0),
                )
                if train_eval.get('report_path'):
                    logger.info("  Report: %s", train_eval.get('report_path'))
            else:
                logger.error("Train evaluation error: %s", train_eval.get('error'))

            if test_samples:
                test_results = process_docs(test_samples, args, task, "Test")
                normalize_result_scores(test_results, task)
                test_eval = evaluate_on_test(
                    test_results,
                    trained_scorer,
                    args,
                    task,
                    output_dir=output_dir,
                    split_name="test",
                )
                stats['test'] = test_eval
                if 'error' not in test_eval:
                    logger.info("Test Results:")
                    logger.info(
                        "  MAE: %.3f",
                        test_eval.get('mae', 0),
                    )
                    logger.info(
                        "  Within 5%%: %.1f%%",
                        test_eval.get('within_5pct', 0),
                    )
                    logger.info(
                        "  Within 10%%: %.1f%%",
                        test_eval.get('within_10pct', 0),
                    )
                    logger.info(
                        "  Evaluated: %s/%s",
                        test_eval.get('n_evaluated', 0),
                        test_eval.get('n_examples', 0),
                    )
                    if test_eval.get('report_path'):
                        logger.info("  Report: %s", test_eval.get('report_path'))
                else:
                    logger.error("Test evaluation error: %s", test_eval.get('error'))
            else:
                logger.info("Skipping test evaluation: no test samples provided")
                stats['test'] = {'processed': 0, 'evaluated': False}
        elif test_samples:
            logger.warning("Skipping train/test evaluation: no trained scorer available")
            test_results = process_docs(test_samples, args, task, "Test")
            normalize_result_scores(test_results, task)
            stats['test'] = {'processed': len(test_results), 'evaluated': False}

        stats['completed_at'] = datetime.now().isoformat()
        stats['success'] = True

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        stats['error'] = str(e)
        stats['success'] = False

    # Save results
    save_results(stats, output_dir)

    return stats


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    args = normalize_judge_optimization_args(args)

    try:
        stats = run_training_pipeline(args)
        return 0 if stats.get('success') else 1
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
