"""ThinkingTrees adapters for the standalone :mod:`treepo` package.

Submodules are imported lazily so that importing one bridge (e.g. ``fno``)
does not eagerly pull in another bridge's heavy dependency chain. The ``markov``
bridge reaches ``src.tree``/``src.preprocessing`` (and through them
``langextract``); the ``fno``/``cardinality`` bridges should not require that
chain merely to be imported.
"""

from __future__ import annotations

from typing import Any

_FNO_EXPORTS = {
    "THINKINGTREES_FNO_FAMILY",
    "HashingEmbeddingClient",
    "build_fno_family",
    "register_fno_family",
}
_MARKOV_EXPORTS = {
    "MARKOV_BENCHMARK",
    "make_markov_trees",
    "register_markov_benchmark",
    "run_markov_benchmark",
}
_CARDINALITY_EXPORTS = {
    "CARDINALITY_BENCHMARK",
    "make_cardinality_documents",
    "register_cardinality_benchmark",
    "run_cardinality_benchmark",
}
_MANIFESTO_EXPORTS = {
    "DEFAULT_MANIFESTO_FINETUNE_ADAPTERS",
    "DEFAULT_MANIFESTO_LEARNING_ADAPTERS",
    "DEFAULT_QSENTENCE_FINETUNE_ADAPTERS",
    "DEFAULT_QSENTENCE_LEARNING_ADAPTERS",
    "build_manifesto_labeled_tree_preferences",
    "build_manifesto_labeled_tree_records",
    "build_manifesto_qsentence_preferences",
    "build_manifesto_qsentence_tree_records",
    "export_manifesto_labeled_tree_finetune_bundle",
    "export_manifesto_qsentence_finetune_adapters",
    "export_manifesto_qsentence_finetune_bundle",
}

__all__ = sorted(
    _FNO_EXPORTS
    | _MARKOV_EXPORTS
    | _CARDINALITY_EXPORTS
    | _MANIFESTO_EXPORTS
    | {"register_treepo_bridges"}
)


def __getattr__(name: str) -> Any:
    if name in _FNO_EXPORTS:
        from src.ctreepo.treepo_bridge import fno

        return getattr(fno, name)
    if name in _MARKOV_EXPORTS:
        from src.ctreepo.treepo_bridge import markov

        return getattr(markov, name)
    if name in _CARDINALITY_EXPORTS:
        from src.ctreepo.treepo_bridge import cardinality

        return getattr(cardinality, name)
    if name in _MANIFESTO_EXPORTS:
        from src.ctreepo.treepo_bridge import manifesto_preferences

        return getattr(manifesto_preferences, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register_treepo_bridges() -> None:
    """Register all ThinkingTrees adapters with the standalone treepo package."""

    from src.ctreepo.treepo_bridge.fno import register_fno_family
    from src.ctreepo.treepo_bridge.markov import register_markov_benchmark
    from src.ctreepo.treepo_bridge.cardinality import register_cardinality_benchmark

    register_fno_family()
    register_markov_benchmark()
    register_cardinality_benchmark()
