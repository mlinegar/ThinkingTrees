"""Cardinality / learned-sketch recovery adapter for the standalone ``treepo``.

Parallels :mod:`src.ctreepo.treepo_bridge.markov`: ThinkingTrees owns the
simulation (``src.tree.cardinality_recovery``), the package owns the small
benchmark/IO contract. This module connects them without moving the simulation
into ``treepo``.

Unlike the Markov benchmark (which reuses the built-in ``oracle`` method), the
cardinality recovery experiment is a self-contained learning sweep
(``run_learning_vs_hll_experiment``) that does not route through the
``oracle``/``fit``/``audit`` method axes. So the runnable entry point executes
the experiment directly and writes artifacts with the same ``treepo.bench.io``
helpers the package runner uses; ``register_cardinality_benchmark`` registers a
discoverability spec so the benchmark shows up in ``list_task_benchmarks()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


CARDINALITY_BENCHMARK = "cardinality"

# Task-config keys forwarded onto ``SimulationConfig`` (kept small and explicit,
# mirroring the Markov bridge's allowed-key list).
_ALLOWED_TASK_CONFIG_KEYS: tuple[str, ...] = (
    "universe_size",
    "min_tokens",
    "max_tokens",
    "leaf_size",
    "zipf_alphas",
    "state_dims",
    "train_docs_grid",
    "train_sizes",
    "n_val",
    "n_test",
    "hidden_dim",
    "n_epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "c3_weight",
    "leaf_weight",
    "idemp_weight",
    "audit_policy",
    "simulation_mode",
    "use_cuda",
    "seed",
    "data_seed",
)

_DEFAULT_TASK_CONFIG: dict[str, Any] = {
    "universe_size": 2048,
    "leaf_size": 32,
    "zipf_alphas": (0.8, 1.0, 1.2),
    "state_dims": (16, 32),
    "train_docs_grid": (128, 256),
    "n_val": 128,
    "n_test": 256,
    "n_epochs": 6,
    "simulation_mode": "latent_proxy_baseline",
    "use_cuda": False,
    "seed": 0,
    "data_seed": 0,
}


def make_cardinality_documents(
    *,
    n_docs: int = 64,
    seed: int = 0,
    universe_size: int = 2048,
    min_tokens: int = 128,
    max_tokens: int = 512,
    leaf_size: int = 32,
    zipf_alphas: Sequence[float] = (0.8, 1.0, 1.2),
) -> list[Any]:
    """Generate ThinkingTrees cardinality documents (data helper).

    Mirrors :func:`make_markov_trees`: a thin, import-light wrapper over
    the ThinkingTrees generator that downstream callers can use without going
    through the full experiment.
    """

    from src.tree.cardinality_recovery import generate_cardinality_documents

    return list(
        generate_cardinality_documents(
            int(n_docs),
            universe_size=int(universe_size),
            min_tokens=int(min_tokens),
            max_tokens=int(max_tokens),
            leaf_size=int(leaf_size),
            zipf_alphas=tuple(zipf_alphas),
            seed=int(seed),
        )
    )


def _build_simulation_config(task_config: Mapping[str, Any]) -> Any:
    from src.tree.cardinality_recovery import SimulationConfig

    kwargs = {k: v for k, v in dict(task_config).items() if k in _ALLOWED_TASK_CONFIG_KEYS}
    for tuple_key in ("zipf_alphas", "state_dims", "train_docs_grid", "train_sizes"):
        if tuple_key in kwargs and kwargs[tuple_key] is not None:
            kwargs[tuple_key] = tuple(kwargs[tuple_key])
    return SimulationConfig(**kwargs)


def register_cardinality_benchmark() -> None:
    """Register the cardinality benchmark with ``treepo`` for discoverability."""

    from treepo.bench.tasks import (
        TaskBenchmarkSpec,
        list_task_benchmarks,
        register_task_benchmark,
    )

    if CARDINALITY_BENCHMARK in set(list_task_benchmarks()):
        return
    register_task_benchmark(
        TaskBenchmarkSpec(
            name=CARDINALITY_BENCHMARK,
            default_method="fit",
            default_scorer="learned_mergeable_sketch",
            supported_scorers=("learned_mergeable_sketch",),
            default_task_config=dict(_DEFAULT_TASK_CONFIG),
            allowed_task_config_keys=_ALLOWED_TASK_CONFIG_KEYS,
            build_method_config=_build_method_config,
        )
    )


def run_cardinality_benchmark(
    *,
    config: Mapping[str, Any] | None = None,
    json_out: str | Path,
    csv_out: str | Path,
    print_json: bool = False,
) -> dict[str, Any]:
    """Run the cardinality recovery sweep and write package-style artifacts.

    Executes ``run_learning_vs_hll_experiment`` directly (the experiment is not
    a method-dispatch task) and writes JSON/CSV using the same
    ``treepo.bench.io`` helpers as :func:`treepo.bench.runner.run_single`.
    """

    import json

    from src.tree.cardinality_recovery import (
        experiment_rows,
        run_learning_vs_hll_experiment,
    )
    from treepo.bench.io import (
        add_runtime_meta,
        atomic_write_text,
        dump_json,
        write_csv_rows,
    )

    task_config = dict(_DEFAULT_TASK_CONFIG)
    task_config.update(dict(config or {}))
    sim_config = _build_simulation_config(task_config)

    summary = run_learning_vs_hll_experiment(sim_config)
    rows = list(experiment_rows(summary.results))
    payload = add_runtime_meta(
        {
            "experiment": CARDINALITY_BENCHMARK,
            "config": task_config,
            "result": json.loads(summary.to_json()),
            "rows": rows,
        }
    )

    json_path = Path(json_out)
    csv_path = Path(csv_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(json_path, dump_json(payload))
    write_csv_rows(csv_path, rows)
    if print_json:
        print(json_path.read_text(encoding="utf-8"))
    return payload


def _build_method_config(
    config: Any,
    task_config: Mapping[str, Any],
    scorer: str,
    output_dir: Path | None,
) -> dict[str, Any]:
    """Carry the resolved sweep config for callers that introspect the spec.

    The runnable path is :func:`run_cardinality_benchmark`; this exists so the
    registered ``TaskBenchmarkSpec`` is self-describing and discoverable.
    """

    method_config: dict[str, Any] = {
        "scorer": str(scorer),
        "task_config": dict(task_config),
    }
    if output_dir is not None:
        method_config["output_dir"] = str(output_dir)
    return method_config


__all__ = [
    "CARDINALITY_BENCHMARK",
    "make_cardinality_documents",
    "register_cardinality_benchmark",
    "run_cardinality_benchmark",
]
