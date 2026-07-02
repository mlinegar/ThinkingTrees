from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from treepo_cdx.backends import StateShapeContract, SupervisionSpec


@dataclass(frozen=True)
class HLLSketchRuntime:
    precision: int = 10
    hash_bits: int = 64
    schedule: str = "balanced"

    def __post_init__(self) -> None:
        if int(self.precision) < 4:
            raise ValueError("HLL precision must be at least 4")
        if int(self.hash_bits) <= int(self.precision) + 1:
            raise ValueError("hash_bits must be greater than precision + 1")
        object.__setattr__(self, "precision", int(self.precision))
        object.__setattr__(self, "hash_bits", int(self.hash_bits))
        object.__setattr__(self, "schedule", str(self.schedule or "balanced"))

    def state_shape_contract(self) -> StateShapeContract:
        return StateShapeContract(
            state_family="hll_registers",
            shape=(1 << int(self.precision),),
            dtype="uint8",
            metadata={
                "precision": int(self.precision),
                "hash_bits": int(self.hash_bits),
                "schedule": self.schedule,
            },
        )

    def supported_supervisions(self) -> tuple[SupervisionSpec, ...]:
        return (
            SupervisionSpec(name="root"),
            SupervisionSpec(name="local_law", requires_oracle=True),
            SupervisionSpec(name="merge", requires_oracle=True),
        )

    def estimate_cardinality(self, token_ids: Sequence[int]) -> float:
        hll = _import_hll()
        cfg = hll.HLLConfig(precision=int(self.precision), hash_bits=int(self.hash_bits))
        return float(hll.HyperLogLogSketch.from_tokens(cfg, list(token_ids)).estimate())

    def merge_estimate(self, leaf_token_lists: Sequence[Sequence[int]]) -> float:
        hll = _import_hll()
        cfg = hll.HLLConfig(precision=int(self.precision), hash_bits=int(self.hash_bits))
        sketches = [hll.HyperLogLogSketch.from_tokens(cfg, list(tokens)) for tokens in leaf_token_lists]
        if not sketches:
            return 0.0
        return float(hll.reduce_hll_sketches(sketches, schedule=self.schedule).estimate())

    def memory_bytes(self) -> float:
        hll = _import_hll()
        cfg = hll.HLLConfig(precision=int(self.precision), hash_bits=int(self.hash_bits))
        return float(hll.HyperLogLogSketch(cfg).memory_bytes)


def hll_fit_summary(
    leaf_token_lists: Sequence[Sequence[int]],
    *,
    precision: int = 10,
    hash_bits: int = 64,
    schedule: str = "balanced",
) -> dict[str, Any]:
    runtime = HLLSketchRuntime(precision=precision, hash_bits=hash_bits, schedule=schedule)
    leaves = tuple(tuple(int(token) for token in leaf) for leaf in leaf_token_lists)
    tokens = tuple(token for leaf in leaves for token in leaf)
    true_cardinality = float(len(set(tokens)))
    estimate = runtime.merge_estimate(leaves)
    abs_error = abs(estimate - true_cardinality)
    return {
        "backend": "hll_native",
        "precision": int(precision),
        "hash_bits": int(hash_bits),
        "schedule": str(schedule),
        "n_leaves": len(leaves),
        "n_tokens": len(tokens),
        "true_cardinality": true_cardinality,
        "estimate": estimate,
        "abs_error": abs_error,
        "rel_error": abs_error / true_cardinality if true_cardinality > 0.0 else 0.0,
        "memory_bytes": runtime.memory_bytes(),
        "capabilities": {
            "state_shape_contract": runtime.state_shape_contract().to_dict(),
            "supported_supervisions": [item.to_dict() for item in runtime.supported_supervisions()],
        },
    }


def _import_hll() -> Any:
    _ensure_monorepo_paths()
    from treepo import hll

    return hll


def _ensure_monorepo_paths() -> None:
    for parent in Path(__file__).resolve().parents:
        repo_root = parent
        treepo_src = repo_root / "treepo" / "src"
        if treepo_src.exists():
            path = str(treepo_src)
            if path not in sys.path:
                sys.path.insert(0, path)
            return


__all__ = ["HLLSketchRuntime", "hll_fit_summary"]
