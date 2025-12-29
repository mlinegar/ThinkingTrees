#!/usr/bin/env python3
"""
Collect oracle-labeled preference data for OPS summarization.

Uses a numeric oracle (e.g., RILE scorer) to determine which summary
better preserves the target information.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_python_path():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


setup_python_path()


def main():
    parser = argparse.ArgumentParser(
        description="Collect oracle-labeled preference data"
    )

    parser.add_argument(
        "--oracle-port", type=int, default=8001,
        help="Port for oracle model server (default: 8001)"
    )
    parser.add_argument(
        "--summarizer-port", type=int, default=8000,
        help="Port for summarizer model server (default: 8000)"
    )
    parser.add_argument(
        "--oracle-model", type=str, default="openai/qwen-30b-thinking",
        help="Model name for oracle LM"
    )
    parser.add_argument(
        "--summarizer-model", type=str, default="openai/qwen-30b-thinking",
        help="Model name for summarizer LM"
    )
    parser.add_argument(
        "--k-candidates", type=int, default=4,
        help="Number of candidate summaries per document"
    )
    parser.add_argument(
        "--temperatures", type=float, nargs="+",
        default=None,
        help="Temperatures for diverse generation (default: from config)"
    )
    parser.add_argument(
        "--tie-margin", type=float, default=5.0,
        help="Tie margin in oracle score units (default: 5.0)"
    )
    parser.add_argument(
        "--max-documents", type=int, default=None,
        help="Maximum documents to process"
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Only use training split"
    )
    parser.add_argument(
        "--law-type", type=str, default="sufficiency",
        choices=["sufficiency", "idempotence", "merge"],
        help="OPS law type for comparison (default: sufficiency)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/preferences"),
        help="Output directory for preference data"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to settings.yaml (default: config/settings.yaml)"
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import dspy
    from src.config.dspy_config import configure_dspy
    from src.config.settings import load_settings
    from src.ops_engine.training_framework.oracle_preference import (
        OraclePreferenceCollector,
        OraclePreferenceConfig,
    )
    from src.ops_engine.training_framework.preference import GenerationConfig
    from src.manifesto.dspy_summarizer import LeafSummarizer
    from src.manifesto.position_oracle import create_rile_scorer
    from src.manifesto.data_loader import ManifestoDataLoader

    settings = load_settings(args.config)
    generation_cfg = settings.get("generation", {})
    summarizer_cfg = generation_cfg.get("summarizer", {})
    oracle_cfg = generation_cfg.get("oracle", {})

    if args.temperatures is None:
        args.temperatures = summarizer_cfg.get(
            "candidate_temperatures", [0.3, 0.5, 0.7, 0.9]
        )
    summarizer_temperature = summarizer_cfg.get("temperature", 0.5)
    summarizer_max_tokens = summarizer_cfg.get("max_tokens", 2048)
    oracle_temperature = oracle_cfg.get("temperature", 0.3)
    oracle_max_tokens = oracle_cfg.get("max_tokens", 2048)

    summarizer_lm = dspy.LM(
        model=args.summarizer_model,
        api_base=f"http://localhost:{args.summarizer_port}/v1",
        api_key="not-needed",
        temperature=summarizer_temperature,
        max_tokens=summarizer_max_tokens,
    )
    oracle_lm = dspy.LM(
        model=args.oracle_model,
        api_base=f"http://localhost:{args.oracle_port}/v1",
        api_key="not-needed",
        temperature=oracle_temperature,
        max_tokens=oracle_max_tokens,
    )

    def use_lm(lm):
        configure_dspy(lm=lm)

    use_lm(oracle_lm)
    rile_scorer = create_rile_scorer()

    def oracle_predict(text: str) -> float:
        current_lm = getattr(dspy.settings, "lm", None)
        try:
            use_lm(oracle_lm)
            return float(rile_scorer.value_extractor(text))
        finally:
            if current_lm is not None:
                configure_dspy(lm=current_lm)

    summarizer = LeafSummarizer(use_cot=True)
    generation_configs = [
        GenerationConfig(temperature=temp, prompt_variant=f"temp_{temp}")
        for temp in args.temperatures[:args.k_candidates]
    ]

    collector = OraclePreferenceCollector(
        summarizer=summarizer,
        oracle_predict=oracle_predict,
        k_candidates=args.k_candidates,
        generation_configs=generation_configs,
        config=OraclePreferenceConfig(tie_margin=args.tie_margin),
    )

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

    for i, sample in enumerate(samples):
        doc_id = sample.get("id", f"doc_{i}")
        doc_text = sample.get("text", "") or sample.get("content", "")
        ground_truth_rile = sample.get("rile", None)

        if not doc_text:
            logger.warning(f"Skipping document {doc_id}: no text")
            continue

        logger.info(f"[{i + 1}/{len(samples)}] Processing {doc_id}...")

        use_lm(summarizer_lm)
        try:
            pairs = collector.collect_pairs_for_example(
                example_id=doc_id,
                original_text=doc_text[:8000],
                rubric=rile_rubric,
                ground_truth_score=ground_truth_rile,
                law_type=args.law_type,
                oracle_name="rile_oracle",
            )
            logger.info(f"  Generated {len(pairs)} preference pairs")
        except Exception as exc:
            logger.error(f"  Error: {exc}")

    dataset = collector.get_dataset()
    stats = collector.get_statistics()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output_dir / f"oracle_preferences_{timestamp}.json"
    dataset.save(output_file)

    stats["collection_config"] = {
        "oracle_port": args.oracle_port,
        "summarizer_port": args.summarizer_port,
        "oracle_model": args.oracle_model,
        "summarizer_model": args.summarizer_model,
        "k_candidates": args.k_candidates,
        "temperatures": args.temperatures,
        "tie_margin": args.tie_margin,
        "law_type": args.law_type,
        "max_documents": args.max_documents,
    }
    stats["total_documents"] = len(samples)

    stats_file = args.output_dir / f"oracle_collection_stats_{timestamp}.json"
    with open(stats_file, "w") as handle:
        json.dump(stats, handle, indent=2)

    logger.info(f"Saved preference pairs to {output_file}")
    logger.info(f"Saved stats to {stats_file}")


if __name__ == "__main__":
    main()
