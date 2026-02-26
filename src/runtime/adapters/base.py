from __future__ import annotations

from typing import Iterable, Protocol

from src.runtime.contracts import NodeContract, ProblemSpec


class BenchmarkAdapter(Protocol):
    def load_split(self, split: str, limit: int | None = None) -> Iterable[ProblemSpec]:
        ...

    def build_contract(self, problem: ProblemSpec) -> NodeContract:
        ...

    def score(self, problem: ProblemSpec, runtime_output: dict) -> dict[str, float]:
        ...

    def primary_metric(self) -> str:
        ...

    def supports_tools(self) -> bool:
        ...

