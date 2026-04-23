from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from treepo.bench.classical_sketches import ClassicalSketchComparisonConfig
from treepo.bench.runner import EXPERIMENT_CLASSICAL_SKETCHES, RunSpec


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).replace(",", " ").split() if x.strip()]


def _parse_labels(text: str) -> List[str]:
    return [x.strip().lower() for x in str(text).replace(",", " ").split() if x.strip()]


CAPACITY_PRESETS = {
    "small": {
        "distinct_lg_k": 8,
        "theta_lg_k": 8,
        "cms_num_buckets": 128,
        "frequent_lg_max_map_size": 7,
        "kll_k": 64,
        "quantiles_k": 64,
        "req_k": 8,
        "tdigest_k": 50,
        "tuple_lg_k": 10,
        "varopt_k": 32,
    },
    "medium": {
        "distinct_lg_k": 10,
        "theta_lg_k": 10,
        "cms_num_buckets": 256,
        "frequent_lg_max_map_size": 8,
        "kll_k": 128,
        "quantiles_k": 128,
        "req_k": 12,
        "tdigest_k": 100,
        "tuple_lg_k": 11,
        "varopt_k": 64,
    },
    "large": {
        "distinct_lg_k": 12,
        "theta_lg_k": 12,
        "cms_num_buckets": 512,
        "frequent_lg_max_map_size": 9,
        "kll_k": 256,
        "quantiles_k": 256,
        "req_k": 16,
        "tdigest_k": 200,
        "tuple_lg_k": 12,
        "varopt_k": 128,
    },
}


def build_classical_sketches_suite(
    *,
    out_root: Path,
    skip_existing: bool,
    seeds: Optional[str] = None,
    leaf_counts: Optional[str] = None,
    capacities: Optional[str] = None,
    execution_backend: str = "unified_g",
    include_learned: bool = False,
    learned_targets: Optional[str] = None,
    learned_variants: Optional[str] = None,
    learned_n_epochs: int = 150,
    learned_n_train: int = 128,
    learned_n_val: int = 48,
) -> List[RunSpec]:
    seed_list = _parse_ints(seeds) if seeds is not None else [0]
    leaf_count_list = _parse_ints(leaf_counts) if leaf_counts is not None else [1, 2, 4, 8, 16]
    capacity_list = _parse_labels(capacities) if capacities is not None else ["small", "medium", "large"]
    learned_target_list = tuple(_parse_labels(learned_targets)) if learned_targets is not None else ("all",)
    learned_variant_list = tuple(_parse_labels(learned_variants)) if learned_variants is not None else ("fg",)
    unknown = [label for label in capacity_list if label not in CAPACITY_PRESETS]
    if unknown:
        valid = ", ".join(sorted(CAPACITY_PRESETS))
        raise ValueError(f"unknown classical-sketch capacity labels {unknown}; expected one of: {valid}")
    specs: List[RunSpec] = []
    output_root = Path(out_root) / "classical_sketches" / "paper"
    base = asdict(
        ClassicalSketchComparisonConfig(
            n_docs=32,
            min_tokens=128,
            max_tokens=512,
            leaf_size=64,
            include_families=("distinct", "frequency", "quantile", "set", "sampling"),
        )
    )
    for capacity in capacity_list:
        preset = CAPACITY_PRESETS[capacity]
        for n_leaves in leaf_count_list:
            for seed in seed_list:
                cfg = dict(base)
                cfg.update(preset)
                cfg["seed"] = int(seed)
                cfg["n_leaves"] = int(n_leaves)
                cfg["capacity_label"] = str(capacity)
                cfg["execution_backend"] = str(execution_backend)
                cfg["include_learned"] = bool(include_learned)
                cfg["learned_targets"] = learned_target_list
                cfg["learned_variants"] = learned_variant_list
                cfg["learned_n_epochs"] = int(learned_n_epochs)
                cfg["learned_n_train"] = int(learned_n_train)
                cfg["learned_n_val"] = int(learned_n_val)
                run_dir = output_root / f"capacity_{capacity}" / f"L_{n_leaves}" / f"seed_{seed}"
                json_out = run_dir / "summary.json"
                csv_out = run_dir / "summary.csv"
                cfg_out = run_dir / "config.yaml"
                if skip_existing and json_out.exists() and csv_out.exists():
                    continue
                specs.append(
                    RunSpec(
                        experiment=EXPERIMENT_CLASSICAL_SKETCHES,
                        config=cfg,
                        json_out=json_out,
                        csv_out=csv_out,
                        config_out=cfg_out,
                    )
                )
    return specs


__all__ = ["CAPACITY_PRESETS", "build_classical_sketches_suite"]
