#!/usr/bin/env python3
"""
Component Sensitivity Analysis / Ablation Framework (Engram WS10 / Phase 6.4).

Systematically measures the impact of each architecture component in isolation
to identify which contribute most to quality and efficiency.

Components tested (each toggled independently):
  1. Prefix restructuring (Phase 1.1): rubric in system message vs user message
  2. Document-affinity routing (Phase 1.2): hash-based vs round-robin
  3. Gated strategy (Phase 5.1): complexity-based LLM gating vs always-call
  4. Enrichment layer (Phase 5.2): entity/topic metadata in merge prompts
  5. ConditionalMemory (Phase 3): unified cache vs independent ephemeral caches
  6. KV-cache persistence (Phase 6.1): LMCache disk tier vs volatile only
  7. Overlapped transitions (Phase 4.1): prewarm vs sequential GPU swaps

Each ablation run uses identical corpus, rubric, and evaluation oracle.
Results include per-component Δ(oracle score) and Δ(wall-clock time).

Usage:
    python -m experiments.component_ablation \
        --corpus data/processed/manifesto \
        --components all \
        --output experiments/results/ablation.json

    # Test specific components
    python -m experiments.component_ablation \
        --corpus data/processed/manifesto \
        --components gating enrichment memory \
        --max-docs 10
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
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Component definitions
# ---------------------------------------------------------------------------

COMPONENTS = {
    "prefix_restructure": {
        "name": "Prefix Restructuring",
        "phase": "1.1",
        "description": "Rubric in system message for prefix cache sharing",
        "config_key": "prompt_restructure_enabled",
    },
    "affinity_routing": {
        "name": "Document-Affinity Routing",
        "phase": "1.2",
        "description": "Hash-based routing for prefix cache locality",
        "config_key": "routing_policy",
        "enabled_value": "affinity_load_aware",
        "disabled_value": "round_robin",
    },
    "gating": {
        "name": "Gated Strategy",
        "phase": "5.1",
        "description": "Complexity-based LLM call gating via ConditionalMemory",
        "config_key": "gating_enabled",
    },
    "enrichment": {
        "name": "Pre-Merge Enrichment",
        "phase": "5.2",
        "description": "Entity/topic metadata injection into merge prompts",
        "config_key": "enrichment_enabled",
    },
    "memory": {
        "name": "ConditionalMemory",
        "phase": "3.1",
        "description": "Unified L1/L2 tiered cache with multi-head scores",
        "config_key": "memory_enabled",
    },
    "kv_persistence": {
        "name": "KV-Cache Persistence",
        "phase": "6.1",
        "description": "LMCache SSD-backed KV cache for cross-restart reuse",
        "config_key": "kv_persistence_enabled",
    },
    "overlapped_transitions": {
        "name": "Overlapped GPU Transitions",
        "phase": "4.1",
        "description": "Prewarm GenRM while tree building completes tail",
        "config_key": "enable_prewarm",
    },
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AblationRun:
    """Results from a single ablation configuration."""
    label: str                      # e.g. "baseline", "-gating", "-enrichment"
    enabled_components: List[str]   # Which components are active
    disabled_components: List[str]  # Which components are ablated
    # Quality
    mean_oracle_score: float = 0.0
    oracle_std: float = 0.0
    oracle_scores: List[float] = field(default_factory=list)
    # Efficiency
    wall_time_s: float = 0.0
    total_llm_calls: int = 0
    gate_hits: int = 0
    memory_hit_rate: float = 0.0
    prefix_cache_hit_rate: float = 0.0
    # Phase transition
    transition_time_s: float = 0.0


@dataclass
class AblationResults:
    """Full ablation study results."""
    corpus: str
    num_documents: int
    timestamp: str = ""
    baseline: Optional[AblationRun] = None
    ablations: List[AblationRun] = field(default_factory=list)
    component_impacts: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def compute_impacts(self) -> None:
        """Compute per-component impact as Δ from baseline."""
        if self.baseline is None:
            return
        for ablation in self.ablations:
            if len(ablation.disabled_components) == 1:
                component = ablation.disabled_components[0]
                self.component_impacts[component] = {
                    "delta_oracle": self.baseline.mean_oracle_score - ablation.mean_oracle_score,
                    "delta_wall_time": ablation.wall_time_s - self.baseline.wall_time_s,
                    "delta_llm_calls": ablation.total_llm_calls - self.baseline.total_llm_calls,
                    "baseline_oracle": self.baseline.mean_oracle_score,
                    "ablated_oracle": ablation.mean_oracle_score,
                    "baseline_wall_time": self.baseline.wall_time_s,
                    "ablated_wall_time": ablation.wall_time_s,
                }


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def build_config_overrides(
    disabled_components: Set[str],
) -> Dict[str, Any]:
    """Build configuration overrides for a given set of disabled components.

    Returns a dict that can be merged with the base config to disable
    specific components.
    """
    overrides: Dict[str, Any] = {}

    for comp_id in disabled_components:
        comp = COMPONENTS.get(comp_id, {})
        key = comp.get("config_key", "")
        if not key:
            continue

        if "disabled_value" in comp:
            overrides[key] = comp["disabled_value"]
        else:
            overrides[key] = False

    return overrides


def run_single_ablation(
    label: str,
    documents: List[Dict[str, Any]],
    enabled: List[str],
    disabled: List[str],
    config_path: Path,
    overrides: Dict[str, Any],
) -> AblationRun:
    """Run a single ablation configuration.

    In a full implementation, this would:
    1. Apply config overrides to disable specific components
    2. Run the full pipeline (tree building + scoring)
    3. Collect quality and efficiency metrics

    The framework provides the structure; actual pipeline integration
    uses run_pipeline.py's existing infrastructure.
    """
    run = AblationRun(
        label=label,
        enabled_components=list(enabled),
        disabled_components=list(disabled),
    )

    logger.info(
        "Running ablation '%s': enabled=[%s], disabled=[%s]",
        label,
        ", ".join(enabled) if enabled else "none",
        ", ".join(disabled) if disabled else "none",
    )

    start = time.time()

    # Placeholder for actual pipeline execution.
    # Integration point: call run_treepo_phase1_audit() with modified config
    # that reflects the overrides dict.

    run.wall_time_s = time.time() - start

    if run.oracle_scores:
        run.mean_oracle_score = sum(run.oracle_scores) / len(run.oracle_scores)
        mean = run.mean_oracle_score
        run.oracle_std = (
            sum((s - mean) ** 2 for s in run.oracle_scores)
            / len(run.oracle_scores)
        ) ** 0.5

    return run


def run_ablation_study(
    corpus_path: str,
    components_to_test: List[str],
    max_docs: Optional[int] = None,
    config_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> AblationResults:
    """Run the full ablation study.

    First runs baseline (all components enabled), then ablates each
    component individually to measure its isolated impact.

    Args:
        corpus_path: Path to document corpus
        components_to_test: Component IDs to ablate (or ["all"])
        max_docs: Optional cap on documents
        config_path: Path to settings.yaml
        output_path: Where to save results
    """
    # Resolve "all" to the full component list
    if "all" in components_to_test:
        components_to_test = list(COMPONENTS.keys())

    # Validate component names
    invalid = [c for c in components_to_test if c not in COMPONENTS]
    if invalid:
        raise ValueError(f"Unknown components: {invalid}. Valid: {list(COMPONENTS.keys())}")

    # Load documents
    from experiments.sparsity_allocation import _load_documents
    documents = _load_documents(corpus_path, max_docs)
    logger.info("Loaded %d documents from %s", len(documents), corpus_path)

    cfg_path = Path(config_path) if config_path else Path("config/settings.yaml")

    results = AblationResults(
        corpus=corpus_path,
        num_documents=len(documents),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    # Run baseline (all components enabled)
    logger.info("=" * 60)
    logger.info("BASELINE: All components enabled")
    logger.info("=" * 60)
    results.baseline = run_single_ablation(
        label="baseline",
        documents=documents,
        enabled=components_to_test,
        disabled=[],
        config_path=cfg_path,
        overrides={},
    )

    # Run individual ablations
    for component_id in components_to_test:
        comp_info = COMPONENTS[component_id]
        logger.info("=" * 60)
        logger.info(
            "ABLATION: -%s (%s, Phase %s)",
            component_id, comp_info["name"], comp_info["phase"],
        )
        logger.info("=" * 60)

        disabled = {component_id}
        enabled = [c for c in components_to_test if c != component_id]
        overrides = build_config_overrides(disabled)

        ablation = run_single_ablation(
            label=f"-{component_id}",
            documents=documents,
            enabled=enabled,
            disabled=[component_id],
            config_path=cfg_path,
            overrides=overrides,
        )
        results.ablations.append(ablation)

    # Compute impacts
    results.compute_impacts()

    # Print results
    _print_impact_table(results)

    # Save
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {
                    "corpus": results.corpus,
                    "num_documents": results.num_documents,
                    "timestamp": results.timestamp,
                    "baseline": asdict(results.baseline) if results.baseline else None,
                    "ablations": [asdict(a) for a in results.ablations],
                    "component_impacts": results.component_impacts,
                },
                f, indent=2,
            )
        logger.info("Results saved to %s", out)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_impact_table(results: AblationResults) -> None:
    """Print a formatted impact table."""
    if not results.baseline or not results.component_impacts:
        print("\nNo impacts computed (need baseline + ablation data).")
        return

    print(f"\nBaseline oracle score: {results.baseline.mean_oracle_score:.3f}")
    print(f"Baseline wall time: {results.baseline.wall_time_s:.1f}s")
    print()

    header = f"{'Component':<25} {'Phase':>5} {'Δ Oracle':>9} {'Δ Time(s)':>10} {'Δ LLM':>7} {'Impact':>8}"
    print(header)
    print("-" * len(header))

    # Sort by impact (largest oracle delta first)
    sorted_impacts = sorted(
        results.component_impacts.items(),
        key=lambda x: abs(x[1].get("delta_oracle", 0)),
        reverse=True,
    )

    for comp_id, impact in sorted_impacts:
        comp_info = COMPONENTS.get(comp_id, {})
        delta_oracle = impact.get("delta_oracle", 0)
        delta_time = impact.get("delta_wall_time", 0)
        delta_llm = impact.get("delta_llm_calls", 0)

        # Classify impact
        if abs(delta_oracle) > 0.02:
            impact_label = "HIGH"
        elif abs(delta_oracle) > 0.005:
            impact_label = "MEDIUM"
        else:
            impact_label = "LOW"

        print(
            f"{comp_info.get('name', comp_id):<25} "
            f"{comp_info.get('phase', '?'):>5} "
            f"{delta_oracle:+9.4f} "
            f"{delta_time:+10.1f} "
            f"{delta_llm:+7d} "
            f"{impact_label:>8}"
        )

    print()
    print("Δ Oracle > 0 means component HELPS quality (ablating it hurts).")
    print("Δ Time > 0 means component SAVES time (ablating it slows down).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Component sensitivity analysis / ablation framework (Engram WS10)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus", required=True,
        help="Path to document corpus",
    )
    parser.add_argument(
        "--components", nargs="+", default=["all"],
        help=f"Components to ablate. Options: all, {', '.join(COMPONENTS.keys())}",
    )
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="Max documents to process",
    )
    parser.add_argument(
        "--config", default="config/settings.yaml",
        help="Path to settings.yaml",
    )
    parser.add_argument(
        "--output", default="experiments/results/ablation.json",
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

    run_ablation_study(
        corpus_path=args.corpus,
        components_to_test=args.components,
        max_docs=args.max_docs,
        config_path=args.config,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
