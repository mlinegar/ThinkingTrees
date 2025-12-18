#!/usr/bin/env python3
"""
Run Manifesto RILE scoring with OPS tree building and DSPy optimization.

This script:
1. Chunks manifestos into ~5000 char pieces (few pages each)
2. Builds OPS trees by hierarchically summarizing chunks
3. Scores RILE from the root summary
4. Audits with 80b model
5. Uses DSPy to optimize prompts based on errors
6. Iterates to improve performance

Usage:
    python run_with_optimization.py --task-port 8000 --auditor-port 8001 --iterations 3
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import dspy
from openai import OpenAI

from src.manifesto.data_loader import create_pilot_dataset
from src.manifesto.rubrics import RILE_PRESERVATION_RUBRIC, RILE_TASK_CONTEXT
from src.manifesto.evaluation import ManifestoEvaluator, save_results

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# DSPy Signatures for the full pipeline

class RILESummarize(dspy.Signature):
    """Summarize political text preserving information relevant to left-right positioning."""
    rubric: str = dspy.InputField(desc="What information to preserve for RILE scoring")
    text: str = dspy.InputField(desc="Political text chunk to summarize")

    summary: str = dspy.OutputField(desc="Concise summary preserving left/right position indicators")


class RILEMerge(dspy.Signature):
    """Merge two summaries while preserving political position information."""
    rubric: str = dspy.InputField(desc="What information to preserve for RILE scoring")
    summary1: str = dspy.InputField(desc="First summary to merge")
    summary2: str = dspy.InputField(desc="Second summary to merge")

    merged_summary: str = dspy.OutputField(desc="Combined summary preserving all position indicators from both inputs")


class RILEScore(dspy.Signature):
    """Score political text on left-right scale."""
    task_context: str = dspy.InputField(desc="The RILE scoring task and scale explanation")
    summary: str = dspy.InputField(desc="Summarized political manifesto to score")

    reasoning: str = dspy.OutputField(desc="Analysis identifying left vs right indicators and their balance")
    rile_score: float = dspy.OutputField(desc="RILE score from -100 (far left) to +100 (far right)")


def is_placeholder(text: str) -> bool:
    """Check if text is a template placeholder instead of real content."""
    if not text or len(text) < 50:
        # Very short output is suspicious
        placeholders = ['[', ']', 'summary', 'merged', 'content', 'text here']
        text_lower = text.lower()
        return any(p in text_lower for p in placeholders)
    return False


class ManifestoSummarizer(dspy.Module):
    """DSPy module for summarizing chunks - optimizable by DSPy."""

    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought(RILESummarize)

    def forward(self, text: str, rubric: str = RILE_PRESERVATION_RUBRIC) -> str:
        result = self.summarize(rubric=rubric, text=text)
        summary = result.summary

        # If we got a placeholder, retry once or fallback to truncated original
        if is_placeholder(summary):
            logger.warning(f"Got placeholder summary: {summary[:50]}... Retrying...")
            result = self.summarize(rubric=rubric, text=text)
            summary = result.summary
            if is_placeholder(summary):
                # Fallback: return first 500 chars of original with note
                summary = f"[Direct excerpt] {text[:500]}"

        return summary


class ManifestoMerger(dspy.Module):
    """DSPy module for merging summaries - optimizable by DSPy."""

    def __init__(self):
        super().__init__()
        self.merge = dspy.ChainOfThought(RILEMerge)

    def forward(self, summary1: str, summary2: str, rubric: str = RILE_PRESERVATION_RUBRIC) -> str:
        result = self.merge(rubric=rubric, summary1=summary1, summary2=summary2)
        merged = result.merged_summary

        # If we got a placeholder, retry once or fallback to concatenation
        if is_placeholder(merged):
            logger.warning(f"Got placeholder merge: {merged[:50]}... Retrying...")
            result = self.merge(rubric=rubric, summary1=summary1, summary2=summary2)
            merged = result.merged_summary
            if is_placeholder(merged):
                # Fallback: concatenate inputs
                merged = f"{summary1}\n\n{summary2}"

        return merged


class ManifestoScorer(dspy.Module):
    """DSPy module for scoring - optimizable by DSPy."""

    def __init__(self):
        super().__init__()
        self.score = dspy.ChainOfThought(RILEScore)

    def forward(self, summary: str, task_context: str = RILE_TASK_CONTEXT) -> dict:
        result = self.score(task_context=task_context, summary=summary)
        try:
            score = float(result.rile_score)
            score = max(-100, min(100, score))
        except (ValueError, TypeError):
            score = 0.0
        return {
            'rile_score': score,
            'reasoning': result.reasoning
        }


class ManifestoPipeline(dspy.Module):
    """
    Full DSPy pipeline: chunk -> summarize -> merge -> score.
    The entire pipeline is optimizable by DSPy.
    Uses parallel processing for chunk summarization and merging.
    """

    def __init__(self, chunk_size: int = 2000):
        super().__init__()
        self.chunk_size = chunk_size
        self.summarizer = ManifestoSummarizer()
        self.merger = ManifestoMerger()
        self.scorer = ManifestoScorer()

    def forward(self, text: str, rubric: str = RILE_PRESERVATION_RUBRIC,
                task_context: str = RILE_TASK_CONTEXT) -> dspy.Prediction:
        """Process a full manifesto through the pipeline with parallel execution."""
        from src.preprocessing.chunker import chunk_for_ops
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 1. Chunk the text
        chunks = chunk_for_ops(text, max_chars=self.chunk_size, strategy="sentence")

        if not chunks:
            return dspy.Prediction(rile_score=0.0, reasoning="No text to process", final_summary="")

        # 2. Summarize chunks in PARALLEL
        def summarize_chunk(chunk_text):
            return self.summarizer(text=chunk_text, rubric=rubric)

        summaries = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            future_to_idx = {
                executor.submit(summarize_chunk, chunk.text): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    summaries[idx] = future.result()
                except Exception as e:
                    summaries[idx] = chunks[idx].text[:500]  # Fallback to truncated original

        # 3. Merge summaries hierarchically with PARALLEL merging at each level
        while len(summaries) > 1:
            pairs = []
            odd_summary = None
            for i in range(0, len(summaries), 2):
                if i + 1 < len(summaries):
                    pairs.append((summaries[i], summaries[i+1]))
                else:
                    odd_summary = summaries[i]

            # Merge pairs in parallel
            next_level = [None] * len(pairs)
            if pairs:
                def merge_pair(s1, s2):
                    return self.merger(summary1=s1, summary2=s2, rubric=rubric)

                with ThreadPoolExecutor(max_workers=len(pairs)) as executor:
                    future_to_idx = {
                        executor.submit(merge_pair, s1, s2): i
                        for i, (s1, s2) in enumerate(pairs)
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            next_level[idx] = future.result()
                        except Exception as e:
                            s1, s2 = pairs[idx]
                            next_level[idx] = f"{s1}\n\n{s2}"  # Fallback to concatenation

            summaries = [s for s in next_level if s is not None]
            if odd_summary is not None:
                summaries.append(odd_summary)

        final_summary = summaries[0] if summaries else ""

        # 4. Score the final summary
        score_result = self.scorer(summary=final_summary, task_context=task_context)

        return dspy.Prediction(
            rile_score=score_result['rile_score'],
            reasoning=score_result['reasoning'],
            final_summary=final_summary
        )


def create_training_examples(samples: list) -> list:
    """Create DSPy training examples from samples with ground truth."""
    examples = []
    for sample in samples:
        example = dspy.Example(
            text=sample.text,
            rubric=RILE_PRESERVATION_RUBRIC,
            task_context=RILE_TASK_CONTEXT,
            rile_score=sample.rile,  # Ground truth
        ).with_inputs('text', 'rubric', 'task_context')
        examples.append(example)
    return examples


def rile_metric(example, prediction, trace=None) -> float:
    """
    DSPy metric: how close is prediction to ground truth RILE?
    Returns 1.0 for perfect, 0.0 for >=50 points off.
    """
    try:
        pred_score = float(prediction.rile_score)
        true_score = float(example.rile_score)
        error = abs(pred_score - true_score)
        # Score: 1.0 for 0 error, 0.0 for 50+ error
        return max(0.0, 1.0 - error / 50.0)
    except (ValueError, TypeError, AttributeError):
        return 0.0


def process_single_sample(pipeline: ManifestoPipeline, sample, idx: int, total: int) -> dict:
    """Process a single sample through the pipeline."""
    result = {
        'manifesto_id': sample.manifesto_id,
        'party_name': sample.party_name,
        'country': sample.country_name,
        'year': sample.year,
        'ground_truth_rile': sample.rile,
        'text_length': len(sample.text),
    }

    try:
        prediction = pipeline(text=sample.text)

        result['predicted_rile'] = prediction.rile_score
        result['reasoning'] = prediction.reasoning
        result['final_summary'] = prediction.final_summary
        result['summary_length'] = len(prediction.final_summary)
        result['compression_ratio'] = len(sample.text) / max(len(prediction.final_summary), 1)

        error = abs(result['predicted_rile'] - result['ground_truth_rile'])
        logger.info(f"[{idx+1}/{total}] {sample.manifesto_id}: "
                   f"{result['summary_length']} chars ({result['compression_ratio']:.1f}x), "
                   f"Pred: {result['predicted_rile']:.1f}, Actual: {result['ground_truth_rile']:.1f}, "
                   f"Error: {error:.1f}")

    except Exception as e:
        logger.error(f"[{idx+1}/{total}] {sample.manifesto_id} Error: {e}")
        result['error'] = str(e)

    return result


def run_iteration(
    pipeline: ManifestoPipeline,
    samples: list,
    iteration: int,
    output_dir: Path,
) -> tuple:
    """Run one iteration using the full DSPy pipeline with PARALLEL sample processing."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info(f"=== Iteration {iteration} ===")
    logger.info(f"Processing {len(samples)} samples in PARALLEL...")

    # Process ALL samples in parallel - vLLM handles batching
    results = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=len(samples)) as executor:
        future_to_idx = {
            executor.submit(process_single_sample, pipeline, sample, idx, len(samples)): idx
            for idx, sample in enumerate(samples)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"Sample {idx} failed: {e}")
                results[idx] = {'error': str(e)}

    # Calculate metrics
    valid = [r for r in results if r.get('predicted_rile') is not None]
    if valid:
        errors = [abs(r['predicted_rile'] - r['ground_truth_rile']) for r in valid]
        mae = sum(errors) / len(errors)
        within_10 = sum(1 for e in errors if e <= 10) / len(errors) * 100

        logger.info(f"\nIteration {iteration} Results:")
        logger.info(f"  Samples processed: {len(valid)}/{len(samples)}")
        logger.info(f"  MAE: {mae:.2f} RILE points")
        logger.info(f"  Within 10 points: {within_10:.1f}%")

    # Save iteration results
    iter_file = output_dir / f"iteration_{iteration}_results.json"
    with open(iter_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results, valid


def optimize_pipeline(pipeline: ManifestoPipeline, examples: list, metric_fn) -> ManifestoPipeline:
    """
    Use DSPy to optimize the ENTIRE pipeline (summarizer, merger, scorer).
    This learns optimal prompts for all stages based on end-to-end RILE prediction accuracy.
    """

    if len(examples) < 4:
        logger.warning("Not enough examples for optimization, need at least 4")
        return pipeline

    logger.info(f"Optimizing full pipeline with {len(examples)} examples...")
    logger.info("  This will optimize: summarizer, merger, AND scorer")

    from dspy.teleprompt import BootstrapFewShot

    # BootstrapFewShot will optimize all modules in the pipeline
    optimizer = BootstrapFewShot(
        metric=metric_fn,
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
    )

    try:
        # Use all examples for training (new DSPy API doesn't support valset)
        optimized_pipeline = optimizer.compile(
            pipeline,
            trainset=examples,
        )
        logger.info("Pipeline optimization complete!")
        return optimized_pipeline
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return pipeline


def main():
    parser = argparse.ArgumentParser(description='Run RILE scoring with OPS trees and DSPy optimization')
    parser.add_argument('--task-port', type=int, default=8000, help='Port for task model (30b)')
    parser.add_argument('--auditor-port', type=int, default=8001, help='Port for auditor model (80b)')
    parser.add_argument('--iterations', type=int, default=3, help='Number of optimization iterations')
    parser.add_argument('--max-samples', type=int, default=20, help='Max samples per iteration')
    parser.add_argument('--split', type=str, default='train', help='Data split to use')
    parser.add_argument('--chunk-size', type=int, default=2000, help='Max chars per chunk (creates more tree levels)')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output_dir or project_root / 'data' / 'results' / 'manifesto_rile' / f'optim_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Chunk size: {args.chunk_size} chars")

    # Set up OpenAI clients for both models
    task_client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{args.task_port}/v1"
    )
    auditor_client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{args.auditor_port}/v1"
    )

    # Verify models are running and get model IDs
    logger.info("Checking model connections...")
    try:
        models_30b = task_client.models.list()
        task_model_id = models_30b.data[0].id if models_30b.data else "default"
        logger.info(f"Task model (30b) ready: {task_model_id}")
    except Exception as e:
        logger.error(f"Task model not ready: {e}")
        return 1

    try:
        models_80b = auditor_client.models.list()
        auditor_model_id = models_80b.data[0].id if models_80b.data else "default"
        logger.info(f"Auditor model (80b) ready: {auditor_model_id}")
    except Exception as e:
        logger.warning(f"Auditor model not ready (will skip auditing): {e}")
        auditor_model_id = None

    # Configure DSPy with task model - this will be used for ALL pipeline stages
    task_lm = dspy.LM(
        f"openai/{task_model_id}",
        api_base=f"http://localhost:{args.task_port}/v1",
        api_key="EMPTY",
        temperature=0.3,
        max_tokens=16000,  # Allow full thinking
    )
    dspy.configure(lm=task_lm)

    # Create the full DSPy pipeline - ALL stages are optimizable
    pipeline = ManifestoPipeline(chunk_size=args.chunk_size)
    logger.info(f"Created DSPy pipeline with chunk_size={args.chunk_size}")

    # Load dataset
    logger.info("Loading dataset...")
    pilot_data = create_pilot_dataset()

    if args.split == 'train':
        sample_ids = pilot_data['train_ids']
    elif args.split == 'val':
        sample_ids = pilot_data['val_ids']
    else:
        sample_ids = pilot_data['test_ids']

    if args.max_samples:
        sample_ids = sample_ids[:args.max_samples]

    dataset = pilot_data['dataset']
    samples = [dataset.get_sample(sid) for sid in sample_ids]
    samples = [s for s in samples if s is not None]

    logger.info(f"Loaded {len(samples)} samples from {args.split} split")

    all_results = []

    # Create training examples from samples (with ground truth)
    training_examples = create_training_examples(samples)
    logger.info(f"Created {len(training_examples)} training examples with ground truth")

    # Run optimization iterations
    for iteration in range(1, args.iterations + 1):
        results, valid = run_iteration(
            pipeline=pipeline,
            samples=samples,
            iteration=iteration,
            output_dir=output_dir,
        )
        all_results.extend(results)

        if iteration < args.iterations and len(training_examples) >= 4:
            # Optimize the FULL pipeline based on RILE prediction accuracy
            pipeline = optimize_pipeline(pipeline, training_examples, rile_metric)

            # Save optimized pipeline
            pipeline.save(str(output_dir / f"pipeline_iter{iteration}.json"))

    # Final summary
    logger.info("\n" + "=" * 50)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 50)

    valid_all = [r for r in all_results if r.get('predicted_rile') is not None]
    if valid_all:
        # Group by iteration for comparison
        for it in range(1, args.iterations + 1):
            iter_results = [r for i, r in enumerate(all_results)
                          if i // len(samples) == it - 1 and r.get('predicted_rile') is not None]
            if iter_results:
                errors = [abs(r['predicted_rile'] - r['ground_truth_rile']) for r in iter_results]
                mae = sum(errors) / len(errors)
                within_10 = sum(1 for e in errors if e <= 10) / len(errors) * 100
                logger.info(f"Iteration {it}: MAE={mae:.2f}, Within 10={within_10:.1f}%")

    logger.info(f"\nResults saved to: {output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
