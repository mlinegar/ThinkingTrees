#!/usr/bin/env python3
"""
Collect Preference Pairs Using Oracle Ground Truth Trees.

This script:
1. Loads pre-generated ground truth trees with oracle scores at all levels
2. Generates candidate summaries for chunks using small model
3. Uses GenRM to compare candidates and create preference pairs
4. Links preferences to ground truth for validation
5. Supports all three OPS laws using tree structure

Workflow:
- Sufficiency: Compare summaries of chunks, use leaf node ground truth
- Idempotence: Re-summarize summaries, use first summary's oracle score
- Merge: Summarize merged chunks, use parent node ground truth

Usage:
    # Collect sufficiency preferences
    python collect_preferences_with_ground_truth.py \
        --ground-truth-dir data/ground_truth \
        --law-type sufficiency \
        --genrm-port 8001 \
        --summarizer-port 8000 \
        --output-dir data/preferences

    # Collect idempotence preferences
    python collect_preferences_with_ground_truth.py \
        --ground-truth-dir data/ground_truth \
        --law-type idempotence \
        --genrm-port 8001 \
        --summarizer-port 8000 \
        --output-dir data/preferences

    # Collect merge preferences
    python collect_preferences_with_ground_truth.py \
        --ground-truth-dir data/ground_truth \
        --law-type merge \
        --genrm-port 8001 \
        --summarizer-port 8000 \
        --output-dir data/preferences
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


def setup_python_path():
    """Add project root to Python path."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


setup_python_path()


def main():
    parser = argparse.ArgumentParser(
        description="Collect preference pairs using oracle ground truth trees"
    )

    # Ground truth configuration
    parser.add_argument(
        "--ground-truth-dir", type=Path, required=True,
        help="Directory containing ground truth trees"
    )

    # Server configuration
    parser.add_argument(
        "--genrm-port", type=int, default=8001,
        help="Port for GenRM server (default: 8001)"
    )
    parser.add_argument(
        "--summarizer-port", type=int, default=8000,
        help="Port for summarizer model server (default: 8000)"
    )
    parser.add_argument(
        "--summarizer-model", type=str, default="openai/qwen-30b-thinking",
        help="Model name for summarizer LM"
    )

    # Generation configuration
    parser.add_argument(
        "--k-candidates", type=int, default=4,
        help="Number of candidate summaries per chunk (default: 4)"
    )
    parser.add_argument(
        "--temperatures", type=float, nargs="+",
        default=None,
        help="Temperatures for diverse generation (default: from config)"
    )

    # OPS law configuration
    parser.add_argument(
        "--law-type", type=str, default="sufficiency",
        choices=["sufficiency", "idempotence", "merge", "all"],
        help="OPS law type for preference collection (default: sufficiency)"
    )

    # Data configuration
    parser.add_argument(
        "--max-trees", type=int, default=None,
        help="Maximum trees to process (default: all)"
    )
    parser.add_argument(
        "--max-chunks-per-tree", type=int, default=None,
        help="Maximum chunks per tree to process (default: all)"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/preferences"),
        help="Output directory for preference data"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to settings.yaml (default: config/settings.yaml)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    args = parser.parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Banner
    print()
    print("=" * 70)
    print("  PREFERENCE COLLECTION WITH GROUND TRUTH")
    print("=" * 70)
    print(f"  Ground Truth Dir:  {args.ground_truth_dir}")
    print(f"  GenRM Port:        {args.genrm_port}")
    print(f"  Summarizer Port:   {args.summarizer_port}")
    print(f"  K Candidates:      {args.k_candidates}")
    print(f"  Law Type:          {args.law_type}")
    print(f"  Output Directory:  {args.output_dir}")
    print("=" * 70)
    print()

    # Import dependencies
    import dspy
    from src.config.dspy_config import configure_dspy
    from src.config.settings import load_settings
    from src.ops_engine.training_framework.genrm_preference import (
        GenRMJudge,
        GenRMPreferenceCollector,
    )
    from src.ops_engine.training_framework.oracle_ground_truth import (
        GroundTruthDataset,
    )
    from src.ops_engine.training_framework.preference import (
        PreferenceDataset,
        PreferencePair,
    )
    from src.manifesto.dspy_summarizer import LeafSummarizer
    from src.manifesto.rubrics import RILE_PRESERVATION_RUBRIC

    settings = load_settings(args.config)
    generation_cfg = settings.get("generation", {})
    summarizer_cfg = generation_cfg.get("summarizer", {})
    judge_cfg = generation_cfg.get("genrm_judge", {})

    if args.temperatures is None:
        args.temperatures = summarizer_cfg.get(
            "candidate_temperatures", [0.3, 0.5, 0.7, 0.9]
        )
    summarizer_temperature = summarizer_cfg.get("temperature", 0.5)
    summarizer_max_tokens = summarizer_cfg.get("max_tokens", 2048)
    judge_temperature = judge_cfg.get("temperature", 0.6)
    judge_top_p = judge_cfg.get("top_p", 0.95)
    judge_max_tokens = judge_cfg.get("max_tokens", 2048)

    # Configure summarizer LM
    logger.info(f"Configuring summarizer on port {args.summarizer_port}...")
    summarizer_lm = dspy.LM(
        model=args.summarizer_model,
        api_base=f"http://localhost:{args.summarizer_port}/v1",
        api_key="not-needed",
        temperature=summarizer_temperature,  # Will be overridden per candidate
        max_tokens=summarizer_max_tokens,
    )
    configure_dspy(lm=summarizer_lm)

    # Create summarizer module
    summarizer = LeafSummarizer(use_cot=True)

    # Create GenRM judge
    logger.info(f"Configuring GenRM judge on port {args.genrm_port}...")
    judge = GenRMJudge(
        base_url=f"http://localhost:{args.genrm_port}/v1",
        model_name="nvidia/Qwen3-Nemotron-235B-A22B-GenRM",
        temperature=judge_temperature,
        top_p=judge_top_p,
        max_tokens=judge_max_tokens,
    )

    # Load ground truth trees
    logger.info(f"Loading ground truth trees from {args.ground_truth_dir}...")
    gt_dataset = GroundTruthDataset.load(args.ground_truth_dir)
    logger.info(f"Loaded {len(gt_dataset)} ground truth trees")

    # Get trees to process
    trees = list(gt_dataset.trees.values())
    if args.max_trees:
        trees = trees[:args.max_trees]

    logger.info(f"Processing {len(trees)} trees")

    # Create preference dataset
    pref_dataset = PreferenceDataset()
    pair_counter = 0

    rubric = RILE_PRESERVATION_RUBRIC

    # Process each tree
    for tree_idx, tree in enumerate(trees):
        logger.info(f"\n[{tree_idx+1}/{len(trees)}] Processing tree: {tree.manifesto_id}")
        logger.info(f"  Tree has {tree.num_chunks} nodes, {tree.num_levels} levels")

        # Determine which nodes to process based on law type
        if args.law_type == "sufficiency" or args.law_type == "all":
            # Process leaf nodes for sufficiency
            nodes_to_process = tree.get_leaves()
            law_type = "sufficiency"
        elif args.law_type == "idempotence":
            # Process leaf nodes for idempotence (will re-summarize)
            nodes_to_process = tree.get_leaves()
            law_type = "idempotence"
        elif args.law_type == "merge":
            # Process merge nodes
            nodes_to_process = tree.get_merge_nodes()
            law_type = "merge"
        else:
            nodes_to_process = tree.get_leaves()
            law_type = "sufficiency"

        if args.max_chunks_per_tree:
            nodes_to_process = nodes_to_process[:args.max_chunks_per_tree]

        logger.info(f"  Processing {len(nodes_to_process)} nodes for {law_type}")

        for node_idx, node in enumerate(nodes_to_process):
            logger.info(f"    [{node_idx+1}/{len(nodes_to_process)}] Node: {node.chunk_id}")

            # Generate k candidate summaries
            candidates = []
            for temp in args.temperatures[:args.k_candidates]:
                try:
                    summarizer_lm.kwargs["temperature"] = temp
                    result = summarizer(content=node.text, rubric=rubric)
                    summary = getattr(result, "summary", str(result))
                    candidates.append((summary, temp))
                except Exception as exc:
                    logger.warning(f"      Candidate generation failed: {exc}")

            if len(candidates) < 2:
                logger.warning(f"      Not enough candidates ({len(candidates)}), skipping")
                continue

            logger.info(f"      Generated {len(candidates)} candidates")

            # For idempotence: re-summarize each candidate
            if law_type == "idempotence":
                re_summarized = []
                for summary, temp in candidates:
                    try:
                        result = summarizer(content=summary, rubric=rubric)
                        re_summary = getattr(result, "summary", str(result))
                        re_summarized.append((summary, re_summary, temp))
                    except Exception as exc:
                        logger.warning(f"      Re-summarization failed: {exc}")

                if len(re_summarized) < 2:
                    logger.warning(f"      Not enough re-summaries, skipping")
                    continue

                candidates = re_summarized
                logger.info(f"      Re-summarized {len(candidates)} candidates")

            # Create pairwise comparisons
            num_pairs = 0
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    if law_type == "idempotence":
                        summary_a, re_summary_a, temp_a = candidates[i]
                        summary_b, re_summary_b, temp_b = candidates[j]
                        # Compare re-summaries
                        comp_a, comp_b = re_summary_a, re_summary_b
                    else:
                        summary_a, temp_a = candidates[i]
                        summary_b, temp_b = candidates[j]
                        comp_a, comp_b = summary_a, summary_b

                    # Random swap for position bias mitigation
                    idx_a, idx_b = i, j
                    swapped = random.random() < 0.5
                    if swapped:
                        comp_a, comp_b = comp_b, comp_a
                        idx_a, idx_b = idx_b, idx_a

                    try:
                        # Use GenRM to compare
                        result = judge.compare(
                            context=rubric,
                            original_text=node.text,
                            summary_a=comp_a,
                            summary_b=comp_b,
                        )

                        # Adjust preferred if we swapped
                        preferred = result.preferred
                        if swapped and preferred != "tie":
                            preferred = "B" if preferred == "A" else "A"

                        pair_counter += 1
                        pair = PreferencePair(
                            pair_id=f"gt_{law_type}_{pair_counter:06d}",
                            source_example_id=node.chunk_id,
                            original_text=node.text,
                            rubric=rubric,
                            ground_truth_score=node.rile_score,
                            law_type=law_type,
                            summary_a=comp_a,
                            summary_b=comp_b,
                            preferred=preferred,
                            reasoning=f"{result.reasoning} (candidates {idx_a},{idx_b})",
                            confidence=result.confidence,
                            score_estimate_a=result.helpfulness_a,
                            score_estimate_b=result.helpfulness_b,
                            judge_model="qwen3-nemotron-genrm",
                            generation_config_a={"temperature": args.temperatures[idx_a]},
                            generation_config_b={"temperature": args.temperatures[idx_b]},
                        )
                        pref_dataset.add_pair(pair)
                        num_pairs += 1

                    except Exception as exc:
                        logger.warning(f"      Comparison failed: {exc}")

            logger.info(f"      Created {num_pairs} preference pairs")

            if (node_idx + 1) % 5 == 0:
                stats = pref_dataset.summary()
                logger.info(f"    Progress: {stats['total_pairs']} total pairs")

    # Get final statistics
    stats = pref_dataset.summary()

    # Save preference dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pref_file = args.output_dir / f"preferences_gt_{args.law_type}_{timestamp}.json"
    pref_dataset.save(pref_file)

    # Save DPO format
    dpo_data = pref_dataset.to_dpo_format()
    dpo_file = args.output_dir / f"dpo_gt_{args.law_type}_{timestamp}.json"
    with open(dpo_file, 'w') as f:
        json.dump(dpo_data, f, indent=2)

    # Save collection statistics
    stats["collection_config"] = {
        "ground_truth_dir": str(args.ground_truth_dir),
        "genrm_port": args.genrm_port,
        "summarizer_port": args.summarizer_port,
        "summarizer_model": args.summarizer_model,
        "k_candidates": args.k_candidates,
        "temperatures": args.temperatures,
        "law_type": args.law_type,
        "max_trees": args.max_trees,
        "max_chunks_per_tree": args.max_chunks_per_tree,
        "seed": args.seed,
    }
    stats["num_trees_processed"] = len(trees)

    stats_file = args.output_dir / f"collection_stats_gt_{args.law_type}_{timestamp}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # Summary
    print()
    print("=" * 70)
    print("  COLLECTION COMPLETE")
    print("=" * 70)
    print(f"  Trees processed:        {len(trees)}")
    print(f"  Total pairs:            {stats['total_pairs']}")
    print(f"  Non-tie pairs:          {stats['non_tie_pairs']}")
    print(f"  Prefer A:               {stats['prefer_a']}")
    print(f"  Prefer B:               {stats['prefer_b']}")
    print(f"  Ties:                   {stats['tie_pairs']}")
    print(f"  Avg confidence:         {stats['avg_confidence']:.2f}")
    print(f"  High confidence (>0.8): {stats['high_confidence_pairs']}")
    print()
    print(f"  Preference file:        {pref_file}")
    print(f"  DPO format:             {dpo_file}")
    print(f"  Statistics:             {stats_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
