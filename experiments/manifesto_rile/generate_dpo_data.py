#!/usr/bin/env python3
"""
Generate DPO training data using a trained OPS comparison module.

This script:
1. Loads manifesto documents
2. Generates candidate summaries with a small model
3. Uses a trained OPS comparison module to choose preferred pairs
4. Writes DPO-format training data
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_python_path() -> None:
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


setup_python_path()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DPO data using a trained OPS comparison module"
    )
    parser.add_argument(
        "--comparison-module",
        type=Path,
        required=True,
        help="Path to trained OPS comparison module",
    )
    parser.add_argument(
        "--summarizer-port",
        type=int,
        default=8000,
        help="Port for summarizer model server",
    )
    parser.add_argument(
        "--judge-port",
        type=int,
        default=8000,
        help="Port for judge model server",
    )
    parser.add_argument(
        "--summarizer-model",
        type=str,
        default="openai/qwen-30b-thinking",
        help="Model name for summarizer LM",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="openai/qwen-30b-thinking",
        help="Model name for judge LM",
    )
    parser.add_argument(
        "--k-candidates",
        type=int,
        default=4,
        help="Number of candidate summaries per document",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=None,
        help="Temperatures for diverse generation (default: from config)",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Maximum documents to process",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only use training split",
    )
    parser.add_argument(
        "--law-type",
        type=str,
        default="sufficiency",
        choices=["sufficiency", "idempotence", "merge"],
        help="OPS law type for comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/dpo"),
        help="Output directory for DPO data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to settings.yaml (default: config/settings.yaml)",
    )

    args = parser.parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.law_type != "sufficiency":
        raise ValueError("DPO generation is only supported for law_type='sufficiency'.")

    import dspy
    from src.config.settings import load_settings
    from src.ops_engine.training_framework.ops_comparison_module import OPSComparisonModule
    from src.ops_engine.training_framework.preference import PreferenceDataset, PreferencePair
    from src.manifesto.dspy_summarizer import LeafSummarizer
    from src.manifesto.data_loader import ManifestoDataLoader

    settings = load_settings(args.config)
    generation_cfg = settings.get("generation", {})
    summarizer_cfg = generation_cfg.get("summarizer", {})
    judge_cfg = generation_cfg.get("comparison_judge", {})

    if args.temperatures is None:
        args.temperatures = summarizer_cfg.get(
            "candidate_temperatures", [0.3, 0.5, 0.7, 0.9]
        )
    summarizer_temperature = summarizer_cfg.get("temperature", 0.5)
    summarizer_max_tokens = summarizer_cfg.get("max_tokens", 2048)
    judge_temperature = judge_cfg.get("temperature", 0.3)
    judge_max_tokens = judge_cfg.get("max_tokens", 2048)

    logger.info("Configuring LMs...")
    summarizer_lm = dspy.LM(
        model=args.summarizer_model,
        api_base=f"http://localhost:{args.summarizer_port}/v1",
        api_key="not-needed",
        temperature=summarizer_temperature,
        max_tokens=summarizer_max_tokens,
    )
    judge_lm = dspy.LM(
        model=args.judge_model,
        api_base=f"http://localhost:{args.judge_port}/v1",
        api_key="not-needed",
        temperature=judge_temperature,
        max_tokens=judge_max_tokens,
    )

    def use_lm(lm):
        dspy.configure(lm=lm)

    summarizer = LeafSummarizer(use_cot=True)

    judge = OPSComparisonModule(use_cot=True)
    judge.load(str(args.comparison_module))

    # Load manifesto data
    logger.info("Loading manifesto data...")
    loader = ManifestoDataLoader()
    train_samples, val_samples, _ = loader.get_temporal_split()

    if args.train_only:
        samples = train_samples
    else:
        samples = train_samples + val_samples

    if args.max_documents:
        samples = samples[:args.max_documents]

    logger.info(f"Processing {len(samples)} documents")

    rile_rubric = """Preserve the political positioning (left-right stance) of the content.

Key information to preserve:
- Left-wing indicators: social welfare, equality, international cooperation, environmental protection
- Right-wing indicators: traditional values, free enterprise, national strength, law and order
- Overall political stance and intensity
- Key policy positions and their framing"""

    dataset = PreferenceDataset()
    pair_counter = 0

    for i, sample in enumerate(samples):
        doc_id = sample.get("id", f"doc_{i}")
        doc_text = sample.get("text", "") or sample.get("content", "")
        ground_truth_rile = sample.get("rile", 0.0)

        if not doc_text:
            logger.warning(f"Skipping document {doc_id}: no text")
            continue

        logger.info(f"[{i + 1}/{len(samples)}] Processing {doc_id}...")

        # Generate candidates
        use_lm(summarizer_lm)
        candidates = []
        for temp in args.temperatures[:args.k_candidates]:
            try:
                summarizer_lm.kwargs["temperature"] = temp
                result = summarizer(content=doc_text[:8000], rubric=rile_rubric)
                summary = getattr(result, "summary", str(result))
                candidates.append(summary)
            except Exception as exc:
                logger.warning(f"Candidate generation failed: {exc}")

        if len(candidates) < 2:
            logger.warning(f"Not enough candidates for {doc_id}")
            continue

        # Compare all pairs
        for a_idx in range(len(candidates)):
            for b_idx in range(a_idx + 1, len(candidates)):
                summary_a = candidates[a_idx]
                summary_b = candidates[b_idx]

                swapped = random.random() < 0.5
                if swapped:
                    summary_a, summary_b = summary_b, summary_a

                use_lm(judge_lm)
                result = judge(
                    law_type=args.law_type,
                    rubric=rile_rubric,
                    original_text=doc_text[:8000],
                    summary_a=summary_a,
                    summary_b=summary_b,
                    ground_truth_score=ground_truth_rile,
                )

                preferred = str(getattr(result, "preferred", "tie"))
                if swapped and preferred != "tie":
                    preferred = "B" if preferred == "A" else "A"

                pair_counter += 1
                dataset.add_pair(PreferencePair(
                    pair_id=f"judge_{pair_counter:06d}",
                    source_example_id=doc_id,
                    original_text=doc_text[:8000],
                    rubric=rile_rubric,
                    ground_truth_score=ground_truth_rile,
                    law_type=args.law_type,
                    summary_a=summary_a if not swapped else summary_b,
                    summary_b=summary_b if not swapped else summary_a,
                    preferred=preferred,
                    reasoning=str(getattr(result, "reasoning", "")),
                    confidence=float(getattr(result, "confidence", 0.5)),
                    judge_model=str(args.comparison_module),
                    generation_config_a={"temperature": args.temperatures[a_idx]},
                    generation_config_b={"temperature": args.temperatures[b_idx]},
                ))

    stats = dataset.summary()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preference_path = args.output_dir / f"preferences_{timestamp}.json"
    dataset.save(preference_path)

    dpo_data = dataset.to_dpo_format(law_type="sufficiency")
    dpo_path = args.output_dir / f"dpo_data_{timestamp}.json"
    with open(dpo_path, "w") as handle:
        json.dump(dpo_data, handle, indent=2)

    stats["collection_config"] = {
        "comparison_module": str(args.comparison_module),
        "summarizer_port": args.summarizer_port,
        "judge_port": args.judge_port,
        "summarizer_model": args.summarizer_model,
        "judge_model": args.judge_model,
        "k_candidates": args.k_candidates,
        "temperatures": args.temperatures,
        "law_type": args.law_type,
        "max_documents": args.max_documents,
    }
    stats["total_documents"] = len(samples)

    stats_path = args.output_dir / f"generation_stats_{timestamp}.json"
    with open(stats_path, "w") as handle:
        json.dump(stats, handle, indent=2)

    logger.info(f"Saved preference pairs to {preference_path}")
    logger.info(f"Saved DPO data to {dpo_path}")
    logger.info(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
