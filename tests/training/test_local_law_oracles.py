from __future__ import annotations

from src.training.local_law_oracles import (
    normalize_local_law_oracle_spec,
    resolve_task_local_law_oracle,
)


def test_normalize_local_law_oracle_spec_treats_disabled_values_as_none() -> None:
    assert normalize_local_law_oracle_spec(None) is None
    assert normalize_local_law_oracle_spec("") is None
    assert normalize_local_law_oracle_spec("off") is None
    assert normalize_local_law_oracle_spec("disabled") is None
    assert normalize_local_law_oracle_spec("task") == "task"


def test_resolve_task_local_law_oracle_uses_task_metadata_and_supported_kwargs() -> None:
    calls: list[dict[str, object]] = []

    class _FakeTask:
        name = "fake_exact_task"

        def describe_local_law_oracle(self):
            return {
                "available": True,
                "exact": True,
                "model_backed": False,
                "kind": "task_oracle_exact",
                "spec": "fake_exact_task:oracle",
            }

        def create_local_law_oracle(self, *, max_tokens: int, temperature: float):
            calls.append(
                {
                    "max_tokens": int(max_tokens),
                    "temperature": float(temperature),
                }
            )

            def _predict(text: str) -> float:
                return float(len(text))

            return _predict

    resolution = resolve_task_local_law_oracle(
        _FakeTask(),
        backend_port=8001,
        backend_model="unused",
        max_tokens=96,
        temperature=0.25,
        strict_parse=True,
    )

    assert resolution is not None
    assert calls == [{"max_tokens": 96, "temperature": 0.25}]
    assert resolution.source_kind == "task_oracle_exact"
    assert resolution.source_spec == "fake_exact_task:oracle"
    assert resolution.metadata["exact"] is True
    assert resolution.predictor("abcd") == 4.0


def test_resolve_task_local_law_oracle_returns_none_when_task_has_no_factory() -> None:
    class _NoOracleTask:
        name = "no_oracle"

    assert resolve_task_local_law_oracle(_NoOracleTask()) is None
