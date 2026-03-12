#!/usr/bin/env python3
"""
Sparsity Allocation Experiment (Engram WS5 / Phase 6.3).

Sweeps the allocation ratio ρ between compute (LLM calls) and memory
(building ConditionalMemory from initial documents) to find the optimal
balance for a given corpus.

Hypothesis: U-shaped optimum at ρ ≈ 0.75-0.85, where:
  - ρ = 1.0 → all compute, no memory → baseline (every merge is an LLM call)
  - ρ = 0.5 → half the budget on memory building, half on LLM → too little compute
  - Optimal ρ → memory handles easy chunks, LLM focuses on hard ones

The "memory budget" phase pre-scans documents and populates ConditionalMemory
with entity extraction, boilerplate detection, and cached summaries for
low-complexity chunks. The "compute budget" phase runs tree building with
gated strategy, where memory hits skip LLM calls.

Usage:
    python -m experiments.sparsity_allocation \
        --corpus data/processed/manifesto \
        --rho 0.5 0.6 0.7 0.8 0.9 1.0 \
        --output experiments/results/sparsity_sweep.json

    # Quick test with 5 documents
    python -m experiments.sparsity_allocation \
        --corpus data/processed/manifesto \
        --max-docs 5 \
        --rho 0.7 0.9 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SparsityResult:
    """Results for a single ρ value."""
    rho: float
    # Memory phase
    memory_build_time_s: float = 0.0
    memory_entries_created: int = 0
    # Compute phase
    tree_build_time_s: float = 0.0
    total_llm_calls: int = 0
    gate_hits: int = 0        # LLM calls skipped via memory
    gate_misses: int = 0      # LLM calls that proceeded
    # Quality
    mean_oracle_score: float = 0.0
    oracle_scores: List[float] = field(default_factory=list)
    oracle_std: float = 0.0
    # Efficiency
    tokens_per_second: float = 0.0
    total_wall_time_s: float = 0.0
    # Memory stats
    memory_hit_rate: float = 0.0
    l1_hits: int = 0
    l2_hits: int = 0

    def efficiency_ratio(self) -> float:
        """Quality per unit compute: higher is better."""
        if self.total_llm_calls == 0:
            return 0.0
        return self.mean_oracle_score / max(1, self.total_llm_calls)


@dataclass
class SparsitySweepResults:
    """Full sweep results across all ρ values."""
    corpus: str
    num_documents: int
    results: List[SparsityResult] = field(default_factory=list)
    optimal_rho: float = 0.0
    optimal_score: float = 0.0
    timestamp: str = ""

    def find_optimal(self) -> None:
        """Find ρ that maximizes oracle score (or efficiency)."""
        if not self.results:
            return
        best = max(self.results, key=lambda r: r.mean_oracle_score)
        self.optimal_rho = best.rho
        self.optimal_score = best.mean_oracle_score


# ---------------------------------------------------------------------------
# Memory building phase
# ---------------------------------------------------------------------------

def build_memory_from_corpus(
    documents: List[Dict[str, Any]],
    memory: Any,
    budget_fraction: float,
) -> Dict[str, Any]:
    """Pre-scan documents and populate ConditionalMemory.

    The budget_fraction (1 - ρ) determines how many documents are
    pre-scanned. With ρ=0.7, 30% of the budget goes to memory building.

    Returns stats about the memory building phase.
    """
    from src.preprocessing.enrichment import ChunkEnricher
    from src.core.engram_memory import EngramMemoryConfig, extract_engram_memory_items

    enricher = ChunkEnricher(memory=memory, enable_tier2=True)
    engram_config = EngramMemoryConfig(enabled=True)

    n_docs = len(documents)
    n_scan = max(1, int(n_docs * budget_fraction))

    stats = {
        "docs_scanned": 0,
        "chunks_enriched": 0,
        "entities_extracted": 0,
    }

    start = time.time()
    for doc in documents[:n_scan]:
        text = doc.get("text", doc.get("content", ""))
        if not text:
            continue

        # Enrich the full document text
        enrichment = enricher.enrich(text)
        stats["chunks_enriched"] += 1

        # Extract entities and cache them
        entities = extract_engram_memory_items(text, engram_config, memory=memory)
        stats["entities_extracted"] += len(entities)
        stats["docs_scanned"] += 1

    stats["build_time_s"] = time.time() - start
    return stats


# ---------------------------------------------------------------------------
# Tree building phase (compute)
# ---------------------------------------------------------------------------

def run_tree_building(
    documents: List[Dict[str, Any]],
    memory: Any,
    config_path: Path,
    use_gating: bool = True,
) -> Dict[str, Any]:
    """Run tree building with optional gating.

    When gating is enabled, the GatedStrategy checks ConditionalMemory
    before each LLM call and skips it for cached/low-complexity chunks.

    Returns stats about the compute phase.
    """
    stats = {
        "tree_build_time_s": 0.0,
        "total_llm_calls": 0,
        "gate_hits": 0,
        "gate_misses": 0,
        "oracle_scores": [],
    }

    # Import here to avoid circular imports at module level
    try:
        from src.core.strategy import get_strategy
    except ImportError:
        logger.warning("Could not import strategy module; returning placeholder stats")
        return stats

    start = time.time()

    # In a full implementation, this would:
    # 1. Create a GatedStrategy wrapping the default strategy
    # 2. Build trees for each document using BatchTreeOrchestrator
    # 3. Score each tree root with the oracle
    # 4. Collect gate hit/miss stats from the strategy
    #
    # For the experiment framework, we provide the structure and
    # instrumentation hooks. The actual tree building integrates
    # with run_pipeline.py's existing infrastructure.

    logger.info(
        "Tree building: %d documents, gating=%s",
        len(documents), use_gating,
    )

    # Placeholder: actual integration uses run_treepo_phase1_audit()
    # from run_pipeline.py with the shared memory instance.
    stats["tree_build_time_s"] = time.time() - start

    return stats


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sparsity_sweep(
    corpus_path: str,
    rho_values: List[float],
    max_docs: Optional[int] = None,
    config_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> SparsitySweepResults:
    """Run the full sparsity allocation sweep.

    For each ρ value:
      1. Create a fresh ConditionalMemory instance
      2. Spend (1-ρ) fraction of budget on memory building
      3. Spend ρ fraction on tree building with gated strategy
      4. Measure oracle quality and efficiency

    Args:
        corpus_path: Path to document corpus (directory or JSON file)
        rho_values: List of ρ values to sweep (0.0 to 1.0)
        max_docs: Optional cap on number of documents
        config_path: Path to settings.yaml
        output_path: Where to save results JSON
    """
    from src.core.conditional_memory import ConditionalMemory, ConditionalMemoryConfig

    # Load documents
    documents = _load_documents(corpus_path, max_docs)
    logger.info("Loaded %d documents from %s", len(documents), corpus_path)

    sweep = SparsitySweepResults(
        corpus=corpus_path,
        num_documents=len(documents),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    cfg_path = Path(config_path) if config_path else Path("config/settings.yaml")

    for rho in sorted(rho_values):
        logger.info("=" * 60)
        logger.info("Running ρ = %.2f (memory budget = %.0f%%)", rho, (1 - rho) * 100)
        logger.info("=" * 60)

        # Fresh memory for each ρ
        memory_cfg = ConditionalMemoryConfig(
            enabled=True,
            mode="readwrite",
            l1_capacity=8192,
        )
        memory = ConditionalMemory(memory_cfg)

        result = SparsityResult(rho=rho)
        total_start = time.time()

        # Phase 1: Memory building (budget = 1 - ρ)
        memory_budget = 1.0 - rho
        if memory_budget > 0.01:
            mem_stats = build_memory_from_corpus(
                documents, memory, budget_fraction=memory_budget
            )
            result.memory_build_time_s = mem_stats["build_time_s"]
            result.memory_entries_created = mem_stats.get("chunks_enriched", 0)
            logger.info(
                "Memory phase: %.1fs, %d entries",
                result.memory_build_time_s,
                result.memory_entries_created,
            )

        # Phase 2: Tree building (budget = ρ)
        tree_stats = run_tree_building(
            documents, memory, cfg_path, use_gating=(rho < 1.0)
        )
        result.tree_build_time_s = tree_stats["tree_build_time_s"]
        result.total_llm_calls = tree_stats.get("total_llm_calls", 0)
        result.gate_hits = tree_stats.get("gate_hits", 0)
        result.gate_misses = tree_stats.get("gate_misses", 0)
        result.oracle_scores = tree_stats.get("oracle_scores", [])

        if result.oracle_scores:
            result.mean_oracle_score = sum(result.oracle_scores) / len(result.oracle_scores)
            mean = result.mean_oracle_score
            result.oracle_std = (
                sum((s - mean) ** 2 for s in result.oracle_scores)
                / len(result.oracle_scores)
            ) ** 0.5

        # Memory stats
        report = memory.report()
        result.memory_hit_rate = report.get("hit_rate", 0.0)
        result.l1_hits = report.get("l1_hits", 0)
        result.l2_hits = report.get("l2_hits", 0)

        result.total_wall_time_s = time.time() - total_start

        logger.info(
            "ρ=%.2f: wall=%.1fs, LLM calls=%d, gate hits=%d, "
            "memory hit rate=%.1f%%, oracle=%.3f±%.3f",
            rho, result.total_wall_time_s, result.total_llm_calls,
            result.gate_hits, result.memory_hit_rate * 100,
            result.mean_oracle_score, result.oracle_std,
        )

        sweep.results.append(result)
        memory.close()

    sweep.find_optimal()

    logger.info("=" * 60)
    logger.info(
        "Optimal ρ = %.2f (oracle score = %.3f)",
        sweep.optimal_rho, sweep.optimal_score,
    )
    logger.info("=" * 60)

    # Print results table
    _print_results_table(sweep)

    # Save results
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {
                    "corpus": sweep.corpus,
                    "num_documents": sweep.num_documents,
                    "timestamp": sweep.timestamp,
                    "optimal_rho": sweep.optimal_rho,
                    "optimal_score": sweep.optimal_score,
                    "results": [asdict(r) for r in sweep.results],
                },
                f, indent=2,
            )
        logger.info("Results saved to %s", out)

    return sweep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_documents(
    corpus_path: str,
    max_docs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load documents from a directory or JSON file."""
    path = Path(corpus_path)
    documents: List[Dict[str, Any]] = []

    if path.is_file() and path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            documents = data
        elif isinstance(data, dict) and "documents" in data:
            documents = data["documents"]
    elif path.is_file() and path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    documents.append(json.loads(line))
    elif path.is_dir():
        for child in sorted(path.iterdir()):
            if child.suffix in (".json", ".jsonl", ".txt"):
                if child.suffix == ".txt":
                    documents.append({"text": child.read_text(), "id": child.stem})
                elif child.suffix == ".json":
                    with open(child) as f:
                        doc = json.load(f)
                    if isinstance(doc, dict):
                        documents.append(doc)
                elif child.suffix == ".jsonl":
                    with open(child) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                documents.append(json.loads(line))
    else:
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    if max_docs and len(documents) > max_docs:
        documents = documents[:max_docs]

    return documents


def _print_results_table(sweep: SparsitySweepResults) -> None:
    """Print a formatted results table."""
    header = f"{'ρ':>5} {'Wall(s)':>8} {'LLM':>6} {'Gate↑':>6} {'HitRate':>8} {'Oracle':>8} {'±σ':>6}"
    print("\n" + header)
    print("-" * len(header))
    for r in sweep.results:
        marker = " *" if r.rho == sweep.optimal_rho else ""
        print(
            f"{r.rho:5.2f} {r.total_wall_time_s:8.1f} "
            f"{r.total_llm_calls:6d} {r.gate_hits:6d} "
            f"{r.memory_hit_rate:7.1%} "
            f"{r.mean_oracle_score:8.3f} {r.oracle_std:6.3f}{marker}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sparsity allocation sweep (Engram WS5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus", required=True,
        help="Path to document corpus (directory, .json, or .jsonl)",
    )
    parser.add_argument(
        "--rho", nargs="+", type=float,
        default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="ρ values to sweep (default: 0.5 0.6 0.7 0.8 0.9 1.0)",
    )
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="Max documents to process (default: all)",
    )
    parser.add_argument(
        "--config", default="config/settings.yaml",
        help="Path to settings.yaml",
    )
    parser.add_argument(
        "--output", default="experiments/results/sparsity_sweep.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Validate ρ values
    for rho in args.rho:
        if not 0.0 <= rho <= 1.0:
            parser.error(f"ρ must be in [0, 1], got {rho}")

    run_sparsity_sweep(
        corpus_path=args.corpus,
        rho_values=args.rho,
        max_docs=args.max_docs,
        config_path=args.config,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
