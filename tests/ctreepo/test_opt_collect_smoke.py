from __future__ import annotations

from typing import Optional, Sequence

from src.ctreepo.opt.collect import collect_pairwise_preferences


class DeterministicGenerator:
    def generate(self, x: float, *, n: int, seed: Optional[int] = None) -> Sequence[float]:
        del n, seed
        return (x, x + 1.0)


def test_collect_pairwise_preferences_smoke() -> None:
    examples = [0.2, 0.8]

    def utility_fn(x: float, a: float) -> float:
        # Higher utility = closer to the oracle target x.
        return -abs(a - x)

    records = collect_pairwise_preferences(
        examples,
        candidate_generator=DeterministicGenerator(),
        utility_fn=utility_fn,
        rubric="keep close",
        seed=123,
    )
    assert len(records) == 2
    assert all(record.preferred == "A" for record in records)

